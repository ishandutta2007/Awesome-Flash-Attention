# Long-Context Transformer Pre-training

## Overview
FlashAttention is the key technology enabling LLM context windows to scale from standard 2k tokens to 128k, 1M, or more. By reducing memory overhead from quadratic $O(N^2)$ to linear $O(N)$ with respect to sequence length, models can process long documents, books, and code repositories during pre-training.

## Core Impact
1. **Memory Compression:** Avoids Out-Of-Memory (OOM) errors during the intermediate attention computation step.
2. **Pre-training Efficiency:** Allows training runs on massive GPU clusters to finish significantly faster, saving compute costs.
3. **Multi-Query / Grouped-Query Attention Support:** Pairs with structural variants to scale context length even further during pre-training.

## Context Length Growth Timeline
```mermaid
timeline
    title Context Length Scaling (Standard Models)
    2018 - 2020 : BERT/GPT-2 (512 - 1024 tokens) : High memory bottlenecks
    2021 - 2022 : GPT-3 (2048 tokens) : Limit of standard attention without severe OOM
    2023 : Llama 2 / MPT (4k - 8k tokens) : Early integration of FlashAttention-1 & 2
    2024 - 2026 : Llama 3 / Gemini (128k - 2M+ tokens) : Powered by FlashAttention architectures
```

## References
- [FlashAttention Paper (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135)


---

[← Back to README](../README.md)
