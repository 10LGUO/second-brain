```markdown
---
title: Zebu (FPGA Chip Simulator)
type: entity
tags: [tool, fpga, simulator, chip-verification, performance-measurement]
created: 2026-04-06
updated: 2026-04-06
sources: [5-kernel-dev.md]
---

# Zebu (FPGA Chip Simulator)

Zebu is an FPGA-based hardware emulation platform used in semiconductor development for pre-tape-out chip verification and early software/operator development. It is produced by Synopsys and is used by chip companies to run real workloads on an FPGA model of their chip design before committing to expensive physical fabrication (tape-out).

## Why It Matters to This Wiki

Zebu is cited as the most accurate method for measuring **compute utilization** of AI operators. By inspecting waveforms — whether circuit signals are at level 1 (active) or level 0 (idle) — compute utilization can be calculated directly as the fraction of time circuits are active. This bypasses the difficulty of estimating utilization analytically from mixed compute types (1D CUDA cores, 2D matrix units, SFUs).

## Key Properties

- **Type:** FPGA-based hardware emulator
- **Use case:** Pre-tape-out (pre-fabrication) software development, operator verification, performance characterization
- **Cost:** Very expensive
- **Capability:** Waveform-level visibility into which circuits are active at any moment
- **Compute utilization formula:** (time with signal level 1) / (total time)

## Limitations / Context

- Availability is restricted to chip companies with the budget and hardware
- Not accessible for standard GPU/CPU kernel developers

## Related Concepts
- [[arithmetic-intensity]]
- [[simd-programming-model]]

## Sources
- [[5-kernel-dev]]
```

---
