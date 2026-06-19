# FlashBlockSparseAttention: Block-Sparse Matrix Optimization

## Overview
FlashBlockSparseAttention is a sparse variant of FlashAttention that limits computation and memory overhead to pre-defined block-sparse structures. Instead of calculating attention across all query-key pairs, it utilizes a block-sparsity mask, skipping tiles that are marked as zero.

## Core Mechanism
1. **Block Sparsity Mask:** A binary matrix indicating which $16 \times 16$ or $64 \times 64$ blocks of the attention matrix are non-zero.
2. **Selective SRAM Loading:** The GPU only loads key-value tiles from HBM into SRAM if the corresponding block in the sparsity mask is active (non-zero).
3. **Sequence Length Scaling:** By reducing the number of blocks to process, it changes the theoretical runtime from quadratic $O(N^2)$ to linear or sub-quadratic depending on the sparsity pattern (e.g., local, strided, or block-diagonal).

## Sparsity Loading Diagram
```mermaid
graph LR
    subgraph HBM [HBM Memory]
        K_Blocks[Key Blocks 1..M]
        V_Blocks[Value Blocks 1..M]
    end

    subgraph Mask [Block-Sparsity Mask]
        Active[Active: Block 1 & 3]
        Inactive[Inactive: Block 2]
    end

    subgraph SRAM [SRAM Buffer]
        K1[K Tile 1]
        K3[K Tile 3]
    end

    K_Blocks -->|Check Mask| Mask
    Mask -->|Only Load Active| SRAM
```

## References
- [FlashAttention Paper (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135)


---

[← Back to README](../README.md)
