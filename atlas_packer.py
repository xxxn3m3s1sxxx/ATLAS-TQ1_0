#!/usr/bin/env python3
"""Atlas Packer - Converts Safetensors to .atlas format"""
import struct, torch, numpy as np
from safetensors import safe_open

# Ternary LUT for 5 values -> byte
def ternary_to_byte(vals):
    """Pack 5 ternary values {-1,0,1} into 1 byte"""
    m = [int(v) + 1 for v in vals[:5]]  # {-1,0,1} -> {0,1,2}
    v = m[0] + m[1]*3 + m[2]*9 + m[3]*27 + m[4]*81
    return min(255, int(v * 255 / 242))

def bitnet_to_ternary(b):
    """Unpack 1 BitNet byte to 4 ternary values"""
    return [(b % 3) - 1, ((b // 3) % 3) - 1, 
            ((b // 9) % 3) - 1, ((b // 27) % 3) - 1]

def pack_tensor(tensor):
    """Pack a weight tensor to TQ1_0"""
    if tensor.dtype == torch.uint8:
        # BitNet format: unpack 4 values per byte, repack to 5 per byte
        data = tensor.numpy().flatten().tolist()
        ternary = []
        for b in data:
            ternary.extend(bitnet_to_ternary(b))
        
        # Pack 5 values per byte
        n_out = (len(ternary) + 4) // 5
        packed = bytearray(n_out)
        for i in range(0, len(ternary), 5):
            chunk = ternary[i:i+5]
            if len(chunk) < 5:
                chunk = chunk + [0] * (5 - len(chunk))
            packed[i // 5] = ternary_to_byte(chunk)
        return bytes(packed), True  # True = TQ1_0
    else:
        # F16 tensor (norms, embeddings)
        return tensor.to(torch.float16).numpy().tobytes(), False

def create_atlas(safetensors_path, output_path, arch_type=2):
    """Create .atlas file from Safetensors"""
    print(f"[ATLAS] Creating {output_path}...")
    
    with safe_open(safetensors_path, framework='pt', device='cpu') as f:
        names = list(f.keys())
        tensors = [(name, f.get_tensor(name)) for name in names]
        
        # Extract metadata from first weight tensor
        for name, t in tensors:
            if t.dtype == torch.uint8:
                shape = list(t.shape)
                hidden_dim = shape[0] * 4  # BitNet expands 4x
                break
        
        with open(output_path, 'wb') as out:
            # Header (64 bytes)
            out.write(b'ATLAS')  # Magic
            out.write(struct.pack('<H', 1))  # Version
            out.write(struct.pack('<B', arch_type))  # Architecture
            out.write(struct.pack('<H', 32))  # n_layers (placeholder)
            out.write(struct.pack('<H', hidden_dim))  # hidden_dim
            out.write(struct.pack('<H', hidden_dim * 4))  # intermediate_dim
            out.write(struct.pack('<B', 32))  # n_heads
            out.write(struct.pack('<B', 8))  # n_kv_heads
            out.write(struct.pack('<I', 128256))  # vocab_size
            out.write(b'\x00' * 48)  # reserved
            
            # Tensor directory (8 bytes per tensor)
            data_start = 64 + len(tensors) * 8
            directory = bytearray(len(tensors) * 8)
            
            # Write tensor data and fill directory
            current_offset = data_start
            for idx, (name, tensor) in enumerate(tensors):
                # Determine type
                ttype = 0 if tensor.dtype == torch.uint8 else (1 if 'norm' in name else 2)
                
                # Pack data
                data_bytes, is_tq1_0 = pack_tensor(tensor)
                if is_tq1_0:
                    # Prepend scale (F16 1.0)
                    scale = struct.pack('<H', 15360)  # FP16: 1.0
                    data_bytes = scale + data_bytes
                
                # Align to 32 bytes
                if current_offset % 32 != 0:
                    padding = 32 - (current_offset % 32)
                    current_offset += padding
                    out.write(b'\x00' * padding)
                
                # Write to directory
                directory[idx*8] = ttype
                struct.pack_into('<Q', directory, idx*8 + 1, current_offset)
                
                # Write tensor data
                out.seek(current_offset)
                out.write(data_bytes)
                current_offset += len(data_bytes)
            
            # Write directory at offset 64
            out.seek(64)
            out.write(bytes(directory))
            
            print(f"[ATLAS] Done! Size: {current_offset / 1024 / 1024 / 1024:.2f} GB")
            print(f"[ATLAS] Tensors: {len(tensors)}")

if __name__ == '__main__':
    # Test with BitNet-2B
    create_atlas(
        r"C:\dam\models\bitnet-b1.58-2B-4T\model.safetensors",
        r"C:\dam\atlas\bitnet-2b.atlas",
        arch_type=2
    )
