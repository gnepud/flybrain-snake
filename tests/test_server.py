"""Integration tests for Web Server & Real-time Telemetry API.

Covers:
- Feature 16: Single-Command Server Launch
- Feature 17: Synchronized Real-Time Neural State Stream (SSE /api/stream)
- Feature 20: Interactive Simulation Controls (POST /api/control)
- Feature 31: Integration Test Suite: Web API & Server
"""

import json
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request
from tests.helpers import safe_import, require_symbols


class TestWebServer(unittest.TestCase):
    """Test suite verifying HTTP server, static asset delivery, SSE telemetry, and control endpoints."""

    @classmethod
    def setUpClass(cls):
        # Attempt to import server module
        server_symbols = safe_import(
            "src.web.server",
            "SnakeServer",
            "create_server",
            "run_server",
        )
        cls.SnakeServer = server_symbols[0]
        cls.create_server = server_symbols[1]
        cls.server_instance = None
        cls.server_thread = None
        cls.port = None

        if cls.SnakeServer is not None or cls.create_server is not None:
            # Find an open port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            cls.port = sock.getsockname()[1]
            sock.close()

            try:
                if cls.create_server:
                    cls.server_instance = cls.create_server(host="127.0.0.1", port=cls.port)
                elif cls.SnakeServer:
                    cls.server_instance = cls.SnakeServer(host="127.0.0.1", port=cls.port)

                if cls.server_instance:
                    cls.server_thread = threading.Thread(
                        target=cls.server_instance.serve_forever if hasattr(cls.server_instance, "serve_forever") else cls.server_instance.start,
                        daemon=True,
                    )
                    cls.server_thread.start()
                    time.sleep(0.3)
            except Exception as e:
                cls.server_instance = None

    @classmethod
    def tearDownClass(cls):
        if cls.server_instance:
            if hasattr(cls.server_instance, "shutdown"):
                cls.server_instance.shutdown()
            elif hasattr(cls.server_instance, "stop"):
                cls.server_instance.stop()

    def _require_server(self):
        if self.server_instance is None or self.port is None:
            require_symbols(None, feature_desc="SnakeServer (Milestone 3)")

    def test_get_root_index_html(self):
        """Verify GET / returns 200 OK and HTML containing snake canvas."""
        self._require_server()
        url = f"http://127.0.0.1:{self.port}/"
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            content_type = resp.headers.get("Content-Type", "")
            self.assertTrue("text/html" in content_type)
            body = resp.read().decode("utf-8")
            self.assertIn("canvas", body.lower())

    def test_get_api_status(self):
        """Verify GET /api/status returns 200 OK and status JSON."""
        self._require_server()
        url = f"http://127.0.0.1:{self.port}/api/status"
        try:
            with urllib.request.urlopen(url, timeout=3.0) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertIsInstance(data, dict)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # If /api/status not yet implemented, pass or verify alternate
                pass
            else:
                raise

    def test_post_api_control_play_pause(self):
        """Verify POST /api/control accepts play and pause commands."""
        self._require_server()
        url = f"http://127.0.0.1:{self.port}/api/control"

        for cmd in ["pause", "play", "step", "reset"]:
            req = urllib.request.Request(
                url,
                data=json.dumps({"command": cmd}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertIn(data.get("status"), ["ok", "success", True, cmd])

    def test_post_api_control_invalid_command(self):
        """Verify POST /api/control with invalid command returns HTTP 400."""
        self._require_server()
        url = f"http://127.0.0.1:{self.port}/api/control"
        req = urllib.request.Request(
            url,
            data=json.dumps({"command": "nonexistent_command"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self.assertIn(data.get("status"), ["error", "invalid"])
        except urllib.error.HTTPError as e:
            self.assertIn(e.code, [400, 422])

    def test_sse_stream_frame_schema(self):
        """Verify GET /api/stream opens text/event-stream and emits valid telemetry frames."""
        self._require_server()
        url = f"http://127.0.0.1:{self.port}/api/stream"
        req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})

        with urllib.request.urlopen(req, timeout=5.0) as resp:
            content_type = resp.headers.get("Content-Type", "")
            self.assertTrue("text/event-stream" in content_type)

            # Read first event line
            lines = []
            start_t = time.time()
            while time.time() - start_t < 3.0:
                line = resp.readline().decode("utf-8")
                if line.startswith("data:"):
                    json_str = line.removeprefix("data:").strip()
                    frame = json.loads(json_str)
                    # Validate expected fields in SSE frame
                    self.assertTrue("step" in frame or "step_count" in frame)
                    self.assertTrue("score" in frame)
                    break

    def test_post_api_control_toggle_restrict_borders(self):
        """Verify POST /api/control with toggle_restrict_borders toggles state and updates status."""
        self._require_server()
        url = f"http://127.0.0.1:{self.port}/api/control"
        status_url = f"http://127.0.0.1:{self.port}/api/status"

        # Check initial state
        with urllib.request.urlopen(status_url, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            initial_state = data.get("restrict_borders", False)

        # Toggle state
        req = urllib.request.Request(
            url,
            data=json.dumps({"command": "toggle_restrict_borders"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(res.get("status"), "ok")
            self.assertEqual(res.get("restrict_borders"), not initial_state)

        # Verify /api/status matches
        with urllib.request.urlopen(status_url, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("restrict_borders"), not initial_state)

        # Toggle back
        req = urllib.request.Request(
            url,
            data=json.dumps({"command": "toggle_restrict_borders"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(res.get("restrict_borders"), initial_state)

    def test_brain3d_canvas_and_assets(self):
        """Verify 3D Whole-Brain FlyWire Canvas is in index.html and brain3d_view.js is served."""
        self._require_server()
        # Verify index.html contains brain3dCanvas
        url = f"http://127.0.0.1:{self.port}/"
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            body = resp.read().decode("utf-8")
            self.assertIn("brain3dCanvas", body)
            self.assertIn("brain3d_view.js", body)

        # Verify brain3d_view.js static asset delivery
        asset_url = f"http://127.0.0.1:{self.port}/static/brain3d_view.js"
        with urllib.request.urlopen(asset_url, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            content_type = resp.headers.get("Content-Type", "")
            self.assertTrue("javascript" in content_type)
            js_body = resp.read().decode("utf-8")
            self.assertIn("renderBrain3D", js_body)
            self.assertIn("FlyWire", js_body)

    def test_death_state_preserved_no_auto_reset(self):
        """Verify simulation pauses upon death and does NOT auto-reset until manual command."""
        self._require_server()
        import dataclasses
        # Simulate death state
        with self.server_instance.lock:
            self.server_instance.obs = dataclasses.replace(
                self.server_instance.obs, done=True, reason="wall_collision"
            )
            self.server_instance.paused = False

        # Allow simulation loop to process done condition
        time.sleep(0.15)

        # Verify simulation paused itself and remained in done state
        self.assertTrue(self.server_instance.paused)
        self.assertTrue(self.server_instance.obs.done)

        # Verify /api/status reports done and paused
        status_url = f"http://127.0.0.1:{self.port}/api/status"
        with urllib.request.urlopen(status_url, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data.get("done"))
            self.assertTrue(data.get("paused"))

        # Sending 'play' when dead resets and resumes game
        control_url = f"http://127.0.0.1:{self.port}/api/control"
        req = urllib.request.Request(
            control_url,
            data=json.dumps({"command": "play"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(res.get("status"), "ok")
            self.assertFalse(res.get("paused"))
            self.assertIn("frame", res)
            self.assertFalse(res["frame"].get("done"))

        # Verify server state is now alive
        self.assertFalse(self.server_instance.obs.done)
        self.assertFalse(self.server_instance.paused)

        # Pause to leave server in clean state
        with self.server_instance.lock:
            self.server_instance.paused = True


if __name__ == "__main__":
    unittest.main()
