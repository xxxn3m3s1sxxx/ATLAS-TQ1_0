#!/usr/bin/env python3
"""ATLAS v2.6.0 SSE Web-Server + Prompt-Caching.

FastAPI/SSE wrapper with single-user prompt caching.
Cache persists KV state across generate calls — sub-second prefill
for follow-up questions sharing the same system prompt.

Usage:
    python atlas_server.py --model path/to/model.atlas
    curl -N http://localhost:8080/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
"""
import asyncio, json, argparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from atlas_infer import AtlasModel

# ─── Session state ────────────────────────────────────────────────────
model: AtlasModel | None = None
infer_lock = asyncio.Lock()

def _load_model(path: str, max_seq_len: int = 4096):
    global model
    m = AtlasModel(path, max_seq_len=max_seq_len)
    # Reset cache on load — clean slate
    m.reset_cache()
    model = m

# ─── Request schemas ───────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    stream: bool = True
    session_id: str | None = None

# ─── Generate helpers ──────────────────────────────────────────────────
def _generate(m: AtlasModel, messages: list[dict], max_new_tokens: int,
              temperature: float, top_k: int, top_p: float,
              repetition_penalty: float) -> str:
    return m.generate_c(
        messages, max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k, top_p=top_p,
        repetition_penalty=repetition_penalty)

async def _stream(m: AtlasModel, messages: list[dict], max_new_tokens: int,
                  temperature: float, top_k: int, top_p: float,
                  repetition_penalty: float):
    """Generate with token-by-token SSE streaming via generate_stream."""
    for token_id in m.generate_stream(
            messages, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty):
        token_text = m._cpp_decode([token_id])
        data = json.dumps({"choices": [{"delta": {"content": token_text}}]})
        yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"

# ─── Routes ────────────────────────────────────────────────────────────
app = FastAPI(title="ATLAS TQ1.0", version="2.6.0",
              description="CPU inference engine for BitNet b1.58 ternary-quantized models")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if model is None:
        raise HTTPException(503, "No model loaded")

    messages = [m.model_dump() for m in req.messages]

    async with infer_lock:
        if not req.stream:
            text = _generate(model, messages, req.max_new_tokens,
                           req.temperature, req.top_k, req.top_p,
                           req.repetition_penalty)
            return {"choices": [{"message": {"content": text}}]}
        else:
            return StreamingResponse(
                _stream(model, messages, req.max_new_tokens,
                       req.temperature, req.top_k, req.top_p,
                       req.repetition_penalty),
                media_type="text/event-stream")

@app.post("/reset")
async def reset():
    """Reset KV cache — starts fresh context."""
    if model is None:
        raise HTTPException(503, "No model loaded")
    async with infer_lock:
        model.reset_cache()
    return {"status": "ok", "message": "cache reset"}

# ─── CLI ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ATLAS SSE Web-Server")
    parser.add_argument("--model", required=True, help="Path to .atlas model file")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--max-seq-len", type=int, default=4096,
                        help="KV cache window size (default: 4096)")
    args = parser.parse_args()

    _load_model(args.model, args.max_seq_len)
    print(f"[ATLAS] Server ready on {args.host}:{args.port}")
    print(f"[ATLAS] Model: {args.model}")
    print(f"[ATLAS] Cache: {args.max_seq_len} tokens")
    print(f"[ATLAS] Lock: single-user (async sequential)")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
