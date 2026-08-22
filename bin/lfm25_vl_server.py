#!/usr/bin/env python3
"""Small local OpenAI-compatible adapter for LiquidAI's MLX vision model."""

from __future__ import annotations

import argparse
import base64
import io
import time
import uuid
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel


def _data_url_image(url: str) -> Image.Image:
    if not url.startswith("data:image/") or "," not in url:
        raise ValueError("Only image data URLs are supported by the local vision service")
    encoded = url.split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    return image.convert("RGB")


def _message_parts(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    normalized: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            normalized.append({"role": message.get("role", "user"), "content": content})
            continue
        parts: list[dict[str, str]] = []
        for item in content or []:
            if item.get("type") == "text":
                parts.append({"type": "text", "text": str(item.get("text", ""))})
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {})
                url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                images.append(_data_url_image(str(url)))
                parts.append({"type": "image"})
        normalized.append({"role": message.get("role", "user"), "content": parts})
    if not images:
        raise ValueError("A vision request must include at least one image_url")
    return normalized, images


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[dict[str, Any]]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.2
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.0


def create_app(model_id: str) -> FastAPI:
    from mlx_vlm import apply_chat_template, generate, load

    model, processor = load(model_id)
    app = FastAPI(title="Hermes LFM Vision", version="1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": model_id}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest) -> dict[str, Any]:
        try:
            messages, images = _message_parts(request.messages)
            prompt = apply_chat_template(
                processor,
                model.config,
                messages,
                add_generation_prompt=True,
                num_images=len(images),
            )
            result = generate(
                model,
                processor,
                prompt,
                images,
                temp=request.temperature if request.temperature is not None else 0.2,
                top_k=request.top_k if request.top_k is not None else 50,
                repetition_penalty=request.repetition_penalty or 1.0,
                max_tokens=min(max(request.max_tokens or 256, 1), 512),
                verbose=False,
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        content = result.text.strip()
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(args.model), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
