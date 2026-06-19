# FlashAttention-XLA

## Overview
FlashAttention-XLA is the adaptation of the FlashAttention algorithm for Google's Accelerated Linear Algebra (XLA) compiler, primarily targeted at Google TPUs (Tensor Processing Units) and JAX/PyTorch-XLA workflows. It translates the GPU-oriented memory tiling and fusion patterns to match XLA compilation paradigms and TPU memory structures.

## Core Mechanism
1. **Pallas Kernels:** Written using JAX's Pallas, a kernel language that allows developers to write custom TPU/GPU kernels directly in Python.
2. **TPU Memory Hierarchy:** TPUs use Vector Memory (VMEM) and Scalar Memory (SMEM) along with High Bandwidth Memory (HBM). FlashAttention-XLA maps Query/Key/Value tiles directly into VMEM.
3. **Loop Fusion:** Instructs the XLA compiler to keep intermediate attention tensors within VMEM/SRAM loops, avoiding compilation rewrites that push tensors back to HBM.

## TPU Execution Flow
```mermaid
graph TD
    subgraph HBM [TPU High Bandwidth Memory]
        Q_hbm[Q Tensor]
        K_hbm[K Tensor]
        V_hbm[V Tensor]
    end

    subgraph VMEM [TPU Vector Memory - SRAM equivalent]
        Q_tile[Q Tile]
        K_tile[K Tile]
        V_tile[V Tile]
        Acc[Accumulator]
    end

    subgraph MXU [Matrix Execution Unit]
        MatMul[MatMul & Softmax fusion]
    end

    Q_hbm -->|DMA Transfer| Q_tile
    K_hbm -->|DMA Transfer| K_tile
    V_hbm -->|DMA Transfer| V_tile
    
    Q_tile & K_tile -->|Compute| MXU
    MXU -->|Partial Updates| Acc
    Acc & V_tile -->|Multiply| Final[Final VMEM Block]
```

## References
- [PyTorch/XLA Github Repository](https://github.com/pytorch/xla)
- [JAX Pallas Documentation](https://github.com/google/jax)


---

[← Back to README](../README.md)
