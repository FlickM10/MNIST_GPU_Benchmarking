# MNIST_GPU_Benchmarking

## 🚀 MNIST CNN Training Race — RTX 5060 Laptop (Blackwell) Beats Kaggle P100!

[![PyTorch](https://img.shields.io/badge/PyTorch-Nightly%20cu128-orange)](https://pytorch.org)
[![Time](https://img.shields.io/badge/Time-17.07s-brightgreen)](https://github.com)

**Date**: January 2026  
**Hardware**: NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell architecture, sm_120)  
**Framework**: PyTorch Nightly (2.11.0.dev + cu128)

## 🏆 Final Results

| Platform              | GPU                        | Total Time (10 epochs) | Avg per Epoch | Test Accuracy |
|-----------------------|----------------------------|------------------------|---------------|---------------|
| **My Laptop**         | RTX 5060 Laptop            | **17.07 seconds**      | 1.71s         | 99.19%        |
| Kaggle                | Tesla P100 (Pascal)        | 22.87 seconds          | 2.29s         | 99.34%        |

**Winner: RTX 5060 Laptop — 25% faster than a 2016 datacenter GPU!** 🔥

[![GPU](https://img.shields.io/badge/GPU-RTX%205060%20Laptop-black)](https://www.nvidia.com)


<img width="2163" height="889" alt="predictions_race_pytorch" src="https://github.com/user-attachments/assets/3d38911e-fe97-4fe5-94b0-7a917f53d8ca" />


<img width="2232" height="768" alt="training_history_pytorch" src="https://github.com/user-attachments/assets/5acd2819-7579-4e89-93f3-fb0d3d5b05f3" />



## Highlights
- Full Blackwell (sm_120) support via PyTorch nightly + CUDA 12.8
- No more compatibility warnings
- Blazing fast training: ~1.7 seconds per epoch
- Clean, reproducible code with proper MNIST mirror fix

## Requirements
Uses PyTorch **nightly** for RTX 50-series support:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
