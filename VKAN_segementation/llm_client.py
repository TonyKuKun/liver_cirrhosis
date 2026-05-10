from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class GemmaClient:
    """Small OpenAI-compatible chat client for Gemma/VLM decisions."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemma-4-31b-it",
        base_url: str | None = None,
        timeout: int = 90,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMMA_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = (base_url or os.getenv("GEMMA_API_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.base_url)

    def chat_json(self, system: str, prompt: str, image_paths: list[str | Path] | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {}

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths or []:
            path = Path(path)
            if not path.exists():
                continue
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return {}

        message = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(message, list):
            message = "".join(part.get("text", "") for part in message if isinstance(part, dict))
        return _extract_json(str(message))


def _extract_json(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}


def ask_for_coarse_plan(client: GemmaClient, patient_name: str, is_post_tips: bool, stats: dict[str, float], previews: list[Path]) -> dict[str, Any]:
    """Ask the model for HU threshold and normalized z/y/x crop bounds."""
    system = (
        "You are assisting portal venous CT vessel extraction. "
        "Return strict JSON only. Favor high recall: keep portal vein, splenic vein, "
        "short SMV, LPV/RPV, compensation veins, and TIPS stent if present."
    )
    prompt = {
        "patient": patient_name,
        "is_post_tips": is_post_tips,
        "volume_stats_hu": stats,
        "request": (
            "Choose an HU threshold [low, high] and a normalized crop box with keys "
            "z, y, x, each [start, end] in 0..1. Post-TIPS should allow brighter stent voxels. "
            "Return JSON: {\"hu_low\": number, \"hu_high\": number, "
            "\"crop\": {\"z\":[a,b], \"y\":[a,b], \"x\":[a,b]}, \"notes\": string}."
        ),
    }
    return client.chat_json(system, json.dumps(prompt, ensure_ascii=True), previews)

