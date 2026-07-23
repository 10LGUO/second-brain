import torch
import math
from torch.functional import F


def attention_forward(Q, K, V, mask=None, dropout_p=0, scale=None):
    # Flash Attention forward pass implementation
    # Q: [batch_size, seq_len, num_heads, head_dim]
    # K: [batch_size, seq_len, num_kv_heads, head_dim]
    # V: [batch_size, seq_len, num_kv_heads, head_dim]
    # mask: attention mask (optional)
    # dropout_p: Dropout probability (optional)
    # scale: optional, default to 1/sqrt(head_dim)
    # Return:
    # output: [batch_size, seq_len, num_heads, head_dim] attention output
    # softmax_lse: [batch_size, num_heads, seq_len] log-sum-exp of the attention matrix

    batch_size, seq_len, num_heads, head_dim = Q.shape
    softmax_lse = 1 / math.sqrt(Q.shape[-1])

    if scale is None:
        scale = 1 / math.sqrt(Q.shape[-1])
    
    Q = Q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)

    attn_matrix = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, T, T]

    # Custom mask when you want to prevent certain position from attending to certrain other positions
    # e.g. decoder, future token should not attend to past tokens.
    # e.g.2. padding mask, in batch, sequence of shorter length maybe padded with dummy tokens. Mask
    # can be used to mask out the dummy tokens.
    if mask is None:
        mask_matrix = torch.full((seq_len, seq_len), float("-inf"), device=Q.device)
        mask_matrix = torch.triu(mask_matrix, diagonal=1)
        attn_matrix = attn_matrix + mask_matrix
    else:
        attn_matrix = attn_matrix + mask

    softmax_out = F.softmax(attn_matrix, dim=-1)

    if dropout_p > 0:
        softmax_out = F.dropout(softmax_out, p=dropout_p)

    output = torch.matmul(softmax_out, V)  # [B, H, T, head_dim]
    output = output.transpose(1, 2)  # [B, T, H, head_dim]

    softmax_lse = torch.logsumexp(attn_matrix, dim=-1)
    return output, softmax_lse

# The tensor runs on whatever devices they where created.
def block_attention_forward_optimized(
    Q, K, V, mask=None, dropout_p=0, scale=None, block_size=256
):
    """
    Optimized block Attention forward pass implementation
    Args:
        Q: [batch_size, seq_len, num_heads, head_dim]
        K: [batch_size, seq_len, num_kv_heads, head_dim]
        V: [batch_size, seq_len, num_kv_heads, head_dim]
        mask: [batch_size, seq_len] attention mask (optional)
        dropout_p: Dropout probability
        scale: optional, default to 1/sqrt(head_dim)
        block_size: block size for block-wise computation
    Return:
        output: [batch_size, seq_len, num_heads, head_dim] attention output
    """
    # Get the shape of the input tensors
    batch_size, seq_len, num_heads, head_dim = Q.shape

    if scale is None:
        scale = 1 / math.sqrt(Q.shape[-1])

    # Reshape [batch_size * num_heads, seq_len, head_dim]
    Q = Q.transpose(1, 2).contiguous().view(batch_size * num_heads, seq_len, head_dim)
    K = K.transpose(1, 2).contiguous().view(batch_size * num_heads, seq_len, head_dim)
    V = V.transpose(1, 2).contiguous().view(batch_size * num_heads, seq_len, head_dim)

    output = torch.zeros_like(Q)

    for i in range(0, seq_len, block_size):
        end_i = min(i + block_size, seq_len)
        q_block = Q[:, i:end_i, :]  # [batch_size * num_heads, block_size, head_dim]
        # Need to compute with entire K because query needs to attend to all keys
        s_block = (
            # Essentially this is a sum over the head dimension so normalized over sqrt(head_dim)
            torch.matmul(q_block, K.transpose(-2, -1)) * scale
        )  # [batch_size * num_heads, block_size, seq_len]
        if mask is not None:
            mask_mat = torch.triu(
                torch.ones(seq_len, seq_len, device=Q.device), diagonal=1
            ).bool()
            # unsqueeze to [1, seq_len, seq_len] then expand (broadcast) to [batch_size * num_heads, seq_len, seq_len]
            mask_mat = mask_mat.unsqueeze(0).expand(batch_size * num_heads, -1, -1)
            mask_block = mask_mat[:, i:end_i, :]
            s_block = s_block.masked_fill(mask_block, float("-inf"))
        softmax_block = F.softmax(s_block, dim=-1)
        if dropout_p > 0 and torch.is_grad_enabled():
            softmax_block = F.dropout(softmax_block, p=dropout_p)
        output_block = torch.bmm(
            softmax_block, V
        )  # [batch_size * num_heads, block_size, head_dim]
        output[:, i:end_i, :] = output_block
    # reshape back to [batch_size, seq_len, num_heads, head_dim]
    output = output.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    return output


def page_attention_forward(
    Q,
    K,
    V,
    page_indices,
    page_start_idx,
    page_end_idx,
    mask=None,
    dropout_p=0.0,
    scale=None,
):
    # Page Attention forward pass
    # Inference only. One token lookup at a time.
    # Q:              [batch_size, q_len, num_heads, head_dim]
    # K:              [num_pages, page_size, num_kv_heads, head_dim] KV cache
    # V:              [num_pages, page_size, num_kv_heads, head_dim] KV cache
    # page_indices:   [batch_size, max_pages]  physical page indices for each sequence
    # page_start_idx: [batch_size]              start slot of the first page for each sequence
    # page_end_idx:   [batch_size]              end slot of the last page for each sequence
    # max_pages:      maximum number of pages
    # dropout_p:      Dropout probability

    batch_size, q_len, num_heads, head_dim = Q.shape
    _, k_seq_len, _, _ = K.shape

    # Q lenth must be one, because page attention is for inference only. One token lookup at a time.
    assert q_len == 1, "Q length must be one for page attention"

    if scale is None:
        scale = 1 / math.sqrt(head_dim)

    Q = Q.transpose(1, 2).contiguous().view(batch_size * num_heads, 1, head_dim)
    K = K.transpose(1, 2).contiguous().view(batch_size * num_heads, k_seq_len, head_dim)
    V = V.transpose(1, 2).contiguous().view(batch_size * num_heads, k_seq_len, head_dim)

    s = (
        torch.bmm(Q, K.transpose(-2, -1)) * scale
    )  # [batch_size * num_heads, 1, k_seq_len]
    if mask is not None:
        mask_mat = torch.triu(
            torch.ones(k_seq_len, k_seq_len, device=Q.device), diagonal=1
        ).bool()  # [k_seq_len, k_seq_len]
        mask_mat = mask_mat.unsqueeze(0).unsqueeze(1)  # [1, 1, k_seq_len, k_seq_len]
        mask_mat = mask_mat[:, :, 0, :]  # [B*H, 1, k_seq_len]
        s = s.masked_fill(mask_mat, float("-inf"))
    softmax_out = F.softmax(s, dim=-1)
    if dropout_p > 0 and torch.is_grad_enabled():
        softmax_out = F.dropout(softmax_out, p=dropout_p)
    # [B*H, 1, seq_len] @ [B*H, seq_len, head_dim] = [B*H, 1, head_dim]
    output = torch.bmm(softmax_out, V)  # [batch_size * num_heads, 1, head_dim]
    output = output.view(batch_size, num_heads, head_dim).transpose(1, 2)
    return output


def page_attention_forward_with_cache(
    Q,K,V,page_indices,page_start_idx,page_end_idx,K_cache=None,V_cache=None,cache_offset=0,mask=None,dropout_p=0.0,scale=None
):
    """
    1. KV cache, avoid recomputing the same keys and value
    2. KV of new token can be added to the daily head token changes.
    3. Memory management is a big issue.
    Args:
        q: [batch_size, q_len, num_heads, head_dim]
        k: [batch_size, k_len, num_heads, head_dim]
        v: [batch_size, v_len, num_heads, head_dim]
        page_indices: [batch_size, max_pages]
        page_start_idx: [batch_size]
        page_end_idx: [batch_size]
        n_lru_heads: number of LRU heads
        dropout_p: Dropout probability
        cache_offset: cache offset
    Return:
        output: [batch_size, q_len, num_heads, head_dim] attention output
        new_k_cache
        new_v_cache
    """
    batch_size, q_len, num_heads, head_dim = Q.shape
    _, k_seq_len, _, _ = K.shape
    assert q_len == 1, "Q length must be one for page attention"
    if scale is None:
        scale = 1 / math.sqrt(head_dim)
    
    if K_cache is None:
        max_cache_len = k_seq_len * 2
        K_cache = torch.zeros(batch_size, num_heads, max_cache_len, head_dim, device=Q.device)
        V_cache = torch.zeros(batch_size, num_heads, max_cache_len, head_dim, device=Q.device)
    K_cache[:, cache_offset:cache_offset+k_seq_len, :, :] = K
    V_cache[:, cache_offset:cache_offset+k_seq_len, :, :] = V
    
    Q = Q.transpose(1, 2).contiguous().view(batch_size * num_heads, 1, head_dim)
    K_cache_reshaped = K_cache.view(batch_size * num_heads, max_cache_len, head_dim)
    V_cache_reshaped = V_cache.view(batch_size * num_heads, max_cache_len, head_dim)
    s = torch.bmm(Q, K_cache_reshaped.transpose(-2, -1)) * scale # [batch_size * num_heads, 1, max_cache_len]
    if mask is not None:
        cache_len = K_cache.size(1)
        causal_mask = torch.tril(torch.ones(cache_len, cache_len, device=Q.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1) # [1, 1, cache_len, cache_len]
        causal_mask = causal_mask[:, :, 0, :] # [B*H, 1, cache_len]
        s = s.masked_fill(causal_mask, float('-inf'))
    softmax_out = F.softmax(s, dim=-1)
    if dropout_p > 0 and torch.is_grad_enabled():
        softmax_out = F.dropout(softmax_out, p=dropout_p)
    output = torch.bmm(softmax_out, V_cache_reshaped) # [batch_size * num_heads, 1, head_dim]
    output = output.view(batch_size, num_heads, head_dim).transpose(1, 2)
    return output, K_cache, V_cache

def create_page_indices(seq_len, page_size = 512):
    """
    Create page indices for the given sequence length
    1. Break large sequence into pages of size page_size.
    2. Each page is handled separately.
    3. Support various page size.
    Args:
        seq_len: sequence length
        page_size: page size
    Return:
        page_indices,
        page_start_idx,
        page_end_idx,
    """
    num_pages = (seq_len + page_size - 1) // page_size
    page_indices = list(range(num_pages))
    page_start_idx = [i * page_size for i in range(num_pages)]
    page_end_idx = [min((i+1) * page_size, seq_len) for i in range(num_pages)]
    return page_indices, page_start_idx, page_end_idx

def test_attention_implementation():
    """
    Test the attention implementation
    1. Attention
    2. Attention optimized
    3. Paged Attention
    4. Paged Attention with cache
    """
    batch_size = 2
    seq_len = 1024
    num_heads = 16
    head_dim = 64
    Q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    K = torch.randn(batch_size, seq_len, num_heads, head_dim)
    V = torch.randn(batch_size, seq_len, num_heads, head_dim)
    mask = torch.ones(batch_size, seq_len, dtype = torch.bool)
    mask[:, seq_len//2:] = 0
    print("testing attention")
    output, softmax_lse = attention_forward(Q, K, V, mask)
    print(f"attention output shape: {output.shape}, softmax_lse shape: {softmax_lse.shape}")
    print("testing block attention optimized")
    output_optimized, softmax_lse_optimized = block_attention_forward_optimized(Q, K, V, mask)
    print(f"block attention optimized output shape: {output_optimized.shape}, softmax_lse_optimized shape: {softmax_lse_optimized.shape}")
    print("testing paged attention")
    page_indices, page_start_idx, page_end_idx = create_page_indices(seq_len)
    output_paged, softmax_lse_paged = page_attention_forward(Q, K, V, page_indices, page_start_idx, page_end_idx)
    print(f"paged attention output shape: {output_paged.shape}, softmax_lse_paged shape: {softmax_lse_paged.shape}")
    print("testing paged attention with cache")
    output_paged_with_cache, new_k_cache, new_v_cache = page_attention_forward_with_cache(Q, K, V, page_indices, page_start_idx, page_end_idx)
    print(f"paged attention with cache output shape: {output_paged_with_cache.shape}, new_k_cache shape: {new_k_cache.shape}, new_v_cache shape: {new_v_cache.shape}")
    print("testing result consistency")
    print(f"attention vs attention optimized: {torch.allclose(output, output_optimized, atol=1e-6)}")
    first_token = output[:, :1, :, :]
    print(f"attention vs paged attention: {torch.allclose(first_token, output_paged, atol=1e-6)}")
    print(f"attention vs paged attention with cache: {torch.allclose(first_token, output_paged_with_cache, atol=1e-6)}")

if __name__ == "__main__":
    test_attention_implementation(None)
