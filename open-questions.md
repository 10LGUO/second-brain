# Open Questions

Questions to verify or research later.

---

## Collective Communication

### All-to-All total network traffic

**Question:** Is the total network traffic for All-to-All N(N-1)×D or (N-1)×D?

**Context:** Each card holds total data D and sends different shards to each other card.
- If each card sends D/N to each of the other (N-1) cards: total = N × (N-1) × D/N = **(N-1)×D**
- If each card sends a full copy D to each other card: total = **N(N-1)×D**

The correct interpretation depends on whether All-to-All assumes the data is pre-sharded (D/N per destination) or each card broadcasts its full D. Need to verify against MPI/NCCL spec or a reference implementation.

**Relevant wiki entries:** [[1-overview-nccl]]
