"""Unit tests for Vercel Serverless Function handler (api/index.py).

Verifies:
- WSGI callable contract: app(environ, start_response)
- GET /api/status returns 200 OK and valid JSON
- GET /api/stream returns 200 OK with text/event-stream Content-Type
- POST /api/control handles play, pause, step, reset, and set_speed
- CORS pre-flight OPTIONS returns 200 OK
- 404 for unknown endpoints and 400 for malformed payloads
"""

import io
import json
import unittest

from api.index import app, handler


class TestVercelApiHandler(unittest.TestCase):
    """Test suite verifying Vercel WSGI API handler behavior."""

    def _call_wsgi(self, path, method="GET", body=None, headers=None):
        status_and_headers = {}

        def start_response(status, response_headers, exc_info=None):
            status_and_headers["status"] = status
            status_and_headers["headers"] = dict(response_headers)

        body_bytes = b""
        if body is not None:
            if isinstance(body, (dict, list)):
                body_bytes = json.dumps(body).encode("utf-8")
            elif isinstance(body, str):
                body_bytes = body.encode("utf-8")
            elif isinstance(body, bytes):
                body_bytes = body

        environ = {
            "REQUEST_METHOD": method,
            "PATH_INFO": path,
            "CONTENT_LENGTH": str(len(body_bytes)),
            "wsgi.input": io.BytesIO(body_bytes),
        }
        if headers:
            for k, v in headers.items():
                environ[f"HTTP_{k.upper().replace('-', '_')}"] = v

        response_iter = app(environ, start_response)
        response_body = b"".join(response_iter)
        return status_and_headers.get("status", ""), status_and_headers.get("headers", {}), response_body

    def test_handler_alias(self):
        """Verify handler is an alias to app for Vercel runtime compatibility."""
        self.assertIs(handler, app)

    def test_options_cors(self):
        """Verify OPTIONS requests return 200 with CORS headers."""
        status, headers, body = self._call_wsgi("/api/control", method="OPTIONS")
        self.assertTrue(status.startswith("200"))
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "*")
        self.assertIn("POST", headers.get("Access-Control-Allow-Methods", ""))

    def test_get_status(self):
        """Verify GET /api/status returns valid status metadata."""
        status, headers, body = self._call_wsgi("/api/status", method="GET")
        self.assertTrue(status.startswith("200"))
        self.assertEqual(headers.get("Content-Type"), "application/json")
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("mode"), "vercel-serverless")
        self.assertIn("running", data)
        self.assertIn("fps", data)

    def test_get_stream(self):
        """Verify GET /api/stream returns SSE headers and initial event."""
        status, headers, body = self._call_wsgi("/api/stream", method="GET")
        self.assertTrue(status.startswith("200"))
        self.assertEqual(headers.get("Content-Type"), "text/event-stream")
        text = body.decode("utf-8")
        self.assertTrue(text.startswith("data:"))
        self.assertIn("serverless", text)

    def test_post_control_commands(self):
        """Verify POST /api/control processes pause, play, step, reset, set_speed."""
        # Pause
        status, _, body = self._call_wsgi("/api/control", method="POST", body={"command": "pause"})
        self.assertTrue(status.startswith("200"))
        self.assertTrue(json.loads(body.decode("utf-8")).get("paused"))

        # Play
        status, _, body = self._call_wsgi("/api/control", method="POST", body={"command": "play"})
        self.assertTrue(status.startswith("200"))
        self.assertFalse(json.loads(body.decode("utf-8")).get("paused"))

        # Step
        status, _, body = self._call_wsgi("/api/control", method="POST", body={"command": "step"})
        self.assertTrue(status.startswith("200"))
        self.assertIn("step", json.loads(body.decode("utf-8")))

        # Speed
        status, _, body = self._call_wsgi("/api/control", method="POST", body={"command": "set_speed", "value": 25})
        self.assertTrue(status.startswith("200"))
        self.assertEqual(json.loads(body.decode("utf-8")).get("fps"), 25)

        # Reset
        status, _, body = self._call_wsgi("/api/control", method="POST", body={"command": "reset"})
        self.assertTrue(status.startswith("200"))
        self.assertEqual(json.loads(body.decode("utf-8")).get("status"), "ok")

    def test_post_control_invalid(self):
        """Verify POST /api/control rejects invalid JSON or unknown commands."""
        status, _, _ = self._call_wsgi("/api/control", method="POST", body="not-json")
        self.assertTrue(status.startswith("400"))

        status, _, _ = self._call_wsgi("/api/control", method="POST", body={"command": "dance"})
        self.assertTrue(status.startswith("400"))

    def test_post_not_found(self):
        """Verify POST to unrecognized path returns 404."""
        status, _, _ = self._call_wsgi("/api/unknown", method="POST", body={"command": "play"})
        self.assertTrue(status.startswith("404"))
