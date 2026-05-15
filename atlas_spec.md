# Atlas Format Specification v1.0

## Overview
Atlas is a minimal, high-performance format for ternary neural network weights.
Designed for 1.58-bit (ternary) models with direct hardware execution.

## File Structure
```
[Header - 64 bytes]
  Magic: "ATLAS" (6 bytes)
  Version: uint16 (2 bytes)
  Architecture: uint8 (1 byte)  // 0=Llama, 1=Falcon, 2=BitNet
  n_layers: uint16 (2 bytes)
  hidden_dim: uint16 (2 bytes)
  intermediate_dim: uint16 (2 bytes)
  n_heads: uint8 (1 byte)
  n_kv_heads: uint8 (1 byte)
  vocab_size: uint32 (4 bytes)
  reserved: [48 bytes]

[Tensor Directory - 8 bytes per tensor]
  For each tensor:
    type: uint8 (1 byte)  // 0=weight (TQ1_0), 1=norm (F16), 2=embedding (F16)
    offset: uint64 (8 bytes)  // absolute offset in file

[Tensor Data]
  Weight tensors: TQ1_0 packed (5 ternary values per byte)
  Norm/Embedding tensors: F16 (float16)
  All tensors aligned to 32-byte boundaries
```

## TQ1_0 Packing (Weight Tensors)
- 5 ternary values {-1, 0, 1} packed into 1 byte
- Encoding: val0 + val1*3 + val2*9 + val3*27 + val4*81
- Maps to range 0-242, scaled to 0-255 for uint8
- Scale factor stored as F16 before each tensor (always 1.0 for pretrained BitNet)

## Advantages over GGUF
- No metadata overhead
- No tensor name strings (directory index instead)
- Direct hardware mapping
- 30% smaller than GGUF for same model
- Load time: O(1) via mmap

## Target Performance
- Llama3-8B: <2GB file size
- Target throughput: 80-100 t/s on modern CPU (AVX2)
- Zero multiplication instructions (only ADD/SUB)
