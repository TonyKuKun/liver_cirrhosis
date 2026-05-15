from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.common import GemmaClient


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return b'{"choices":[{"message":{"content":"{\\"ok\\":true}"}}]}'


class DeepSeekClientTests(unittest.TestCase):
    def test_deepseek_chat_payload_uses_text_content_not_image_parts(self) -> None:
        requests = []

        def fake_urlopen(req, timeout=None):
            requests.append(req)
            return _FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "preview.png"
            image.write_bytes(b"png")
            client = GemmaClient(api_key="test", model="deepseek-v4-pro", base_url="https://api.deepseek.com")
            with patch("urllib.request.urlopen", fake_urlopen):
                result = client.chat_json("Return JSON only.", '{"patient":"case"}', [image])

        payload = json.loads(requests[0].data.decode("utf-8"))
        self.assertEqual(result, {"ok": True})
        self.assertEqual(payload["messages"][1]["content"], '{"patient":"case"}')
        self.assertEqual(payload["model"], "deepseek-v4-pro")


if __name__ == "__main__":
    unittest.main()
