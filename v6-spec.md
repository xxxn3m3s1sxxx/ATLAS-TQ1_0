# v6 — Embedded Binary Tokenizer

## Motivation

Remove Python tokenizer dependency entirely. Single FFI call for encode/decode,
no `transformers`/`tokenizers` Python packages needed at runtime.

## File Layout

```
[64-byte header]
[tensor directory: 12 × n_tensors bytes]
[name_block: variable]
[tensor_data: ...]
[tokenizer_json_block: variable]     ← v5 compat, raw tokenizer.json
[tokenizer_binary_block: variable]   ← NEW: pre-built C++ structures
```

### Header Changes (bytes 56–63)

v5 header bytes 56–63 are unused (`name_block_size` at 56–59, `tensor_count` at 60–63).
v6 repurposes 60–63 and adds:

| Offset | Size | Field |
|--------|------|-------|
| 56 | 4 | tokenizer_binary_size (0 = no binary block) |
| 60 | 4 | tokenizer_binary_offset (0 = no binary block) |

Bytes 29–36 remain as v5 (`tokenizer_size` at 29–32, `tokenizer_offset` at 33–36)
for the JSON fallback block.

## Tokenizer Binary Block

### Block Header

```c
struct tokenizer_binary_header_t {
    uint32_t magic;              // 0x544F4B42 = "TOKB"
    uint32_t version;            // 1
    uint32_t vocab_size;         // typically 131072
    uint32_t max_token_length;   // longest decoded token in bytes
    uint32_t special_count;      // number of special tokens (below)
    uint32_t flags;              // reserved (0)
    // Decoder section
    uint64_t offset_offsets;     // byte offset to offsets[]
    uint64_t offset_lengths;     // byte offset to lengths[]
    uint64_t offset_pool;        // byte offset to string_pool
    uint64_t pool_size;          // total size of string_pool
    // Encoder section
    uint64_t offset_merge_left;  // byte offset to merge_left[]
    uint64_t offset_merge_right; // byte offset to merge_right[]
    uint64_t offset_merge_rank;  // byte offset to merge_rank[]
    uint64_t offset_byte_enc;    // byte offset to byte_encoder[256]
    uint64_t offset_byte_dec;    // byte offset to byte_decoder[256]
    // Special tokens
    uint64_t offset_special;     // byte offset to special_tokens[]
    // Padding to 128 bytes
    uint8_t  reserved[16];
};
// Total: 128 bytes
```

### Decoder Section

Three contiguous arrays:

```c
uint32_t offsets[vocab_size];    // byte offset into string_pool
uint16_t lengths[vocab_size];    // byte length of token string
char     pool[pool_size];        // concatenated UTF-8 token strings
```

Decode: `pool + offsets[id]` → `memcpy(dest, pool + offsets[id], lengths[id])`

**6 bytes per token** (4 + 2), ~0.8 MB for 131k vocab.

### Encoder Section

```c
struct merge_entry_t {
    uint32_t left;   // token ID of left child (0xFFFFFFFF = base token)
    uint32_t right;  // token ID of right child
    uint32_t rank;   // merge priority (0 = base token, higher = merged later)
};
```

Or as three parallel arrays (same data, better cache for rank scanning):

```c
uint32_t merge_left[vocab_size];   // 0xFFFFFFFF = base token
uint32_t merge_right[vocab_size];
uint32_t merge_rank[vocab_size];   // 0 for base tokens
```

**12 bytes per token**, ~1.6 MB for 131k vocab.

Plus byte-level LUTs:

```c
uint16_t byte_encoder[256];   // byte value → base token ID (0xFFF = unmapped)
uint16_t byte_decoder[256];   // byte value after BPE decode → original byte
// Total: 1 KB
```

### Special Tokens

Stored as compact array at end of binary block:

```c
struct special_token_t {
    uint32_t id;
    uint16_t type;   // 0=EOS, 1=BOS, 2=PAD, 3=UNK, 4=MASK, 5=SEP, 6=CLS
    uint16_t length;
    // followed by `length` bytes of UTF-8 string
};

// Or flat: indices indexed by type
struct special_tokens_t {
    uint32_t eos_id;   // 0xFFFFFFFF = none
    uint32_t bos_id;
    uint32_t pad_id;
    uint32_t unk_id;
    uint32_t mask_id;
    uint32_t sep_id;
    uint32_t cls_id;
};
// Special token strings are also in the main pool
```

Special tokens live in the main string pool AND are indexed by type in this
mini-array. The encoder checks `unk_id` for unknown byte sequences. The decoder
can special-case `eos_id` to stop generation in `atlas_generate`.

## Packer Changes (`atlas_packer.py`)

After building the tensor data and JSON tokenizer block, the packer:

1. Loads `tokenizer.json` via Python
2. Extracts vocabulary (sorted by ID), merges, byte encoder/decoder
3. Builds the string pool (concatenated, index by vocab ID)
4. Writes `offsets[]`, `lengths[]`, `pool`
5. Writes `merge_left[]`, `merge_right[]`, `merge_rank[]`
6. Writes `byte_encoder[256]`, `byte_decoder[256]`
7. Detects special tokens by scanning token strings (e.g. `<|endoftext|>`)
8. Writes `special_tokens_t` block
9. Appends binary block to `.atlas` file, updates header

## C++ API (`atlas_api.cpp`)

```c
// v6 new API
int atlas_tokenizer_encode(void* model, const char* text, int text_len,
                           uint32_t* out_tokens, int max_tokens);
int atlas_tokenizer_decode(void* model, const uint32_t* tokens, int n_tokens,
                           char* out_text, int max_out);
```

Called from Python ctypes with zero extra overhead:

```python
n = dll.atlas_tokenizer_encode(m.model_ptr, text.encode(), len(text),
                                out_buf, max_len)
```

The `atlas_load` function detects v6 by checking `tokenizer_binary_offset != 0`.
If present, it maps the binary block directly. Otherwise falls back to JSON
tokenizer via Python (v5 compat).

## BPE Encoding Algorithm

```
encode(text):
  bytes = utf8_encode(text)
  tokens = []
  for b in bytes:
    id = byte_encoder[b]
    if id == 0xFFFF: tokens.append(unk_id)
    else: tokens.append(id)

  loop:
    best_rank = INF
    best_pos = -1
    for i = 0 to len(tokens)-2:
      pair = hash(tokens[i], tokens[i+1])
      r = merge_rank[pair]    // direct array lookup
      if r > 0 and r < best_rank:
        best_rank = r
        best_pos = i
    if best_pos == -1: break

    // Merge: replace tokens[best_pos:best_pos+2] with merge_entry[pair]
    // (merge_entry[pair].id = merge_target_id, stored implicitly by position)
    merged_id = pair             // pair IS the merged token ID (BPE invariant)
    replace tokens[best_pos] = merged_id
    remove tokens[best_pos+1]

  return tokens
```

Key property: In BPE, `merge(a, b) → c` where `c` is always a token ID that
didn't exist before, but its index in the merge table IS `c`. So
`merge_left[c] = a`, `merge_right[c] = b`, and `merge_rank[c] > 0`.
This means `pair = (left << 16) | right` is NOT needed — the merged token's
index in the merge arrays is the token ID.

## Memory Budget

| Section | Size |
|---------|------|
| Block header | 128 B |
| offsets[131k], lengths[131k] | 786 KB |
| string_pool (avg 6B/token) | ~770 KB |
| merge_left/right/rank[131k] | 1.57 MB |
| byte_enc/dec[256] | 1 KB |
| special tokens | < 256 B |
| **Total** | **~3.1 MB** |

Well within any reasonable budget. Added to `.atlas` file size:
- 1B: +3.1 MB → 1.22 GB (negligible)
- 10B: +3.1 MB → 3.28 GB (0.09% increase)

## Implementation Order

1. `atlas_packer.py`: Add binary tokenizer block writer
2. `atlas_ffi.h`: Add v6 header fields + new C API
3. `atlas_api.cpp`: Implement `atlas_tokenizer_encode`, `atlas_tokenizer_decode`
4. `atlas_infer.py`: Update `AtlasModel.__init__` to use C++ tokenizer, remove
   `PreTrainedTokenizerFast` dependency
5. `atlas_infer.py`: Simplify `generate_c` — no more Python tokenizer calls
