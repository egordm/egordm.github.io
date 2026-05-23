---
title: "Optimizing SAMURAI"
description: "Taking a video segmentation model from 1.3 FPS to real-time. ONNX export, TensorRT, CoreML, and Rust."
---

A practical engineering series about optimizing the SAMURAI/EfficientTAM video segmentation pipeline for real-time performance. From research-grade PyTorch code that barely runs at 1 FPS, through ONNX export, NVIDIA TensorRT, Apple Silicon CoreML, and ultimately a Rust runtime.

## Posts in this series

1. **[[blog/2026-05-16-optimizing-samurai-part-1|Optimizing PyTorch Models for Production: ONNX Export with SAM 2]]** - Getting the model into ONNX, which sounds simple until you try
2. **[[blog/2026-05-17-optimizing-samurai-part-2|TensorRT Optimization: 5x Faster Inference with FP16 Precision]]** - From 80 FPS to 97 FPS on RTX 4090, including a deep dive into a TensorRT fusion bug
3. **[[blog/2026-05-18-optimizing-samurai-part-3|CoreML Deployment on Apple Silicon: Real-Time Vision Models]]** - Running on Apple Silicon: from CoreML being slower than CPU to hitting 28 FPS
4. *Part 4: Rust Runtime* - Coming soon
