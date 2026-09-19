"""Vercel Serverless Function Handler (WSGI) for FlyBrain Snake API.

Provides:
- GET /api/status: Returns environment and simulation status
- POST /api/control: Handles simulation controls (play, pause, step, reset, set_speed)
- GET /api/stream: Serverless telemetry stream response
"""

import json
from urllib.parse import parse_qs

# Lightweight in-memory state for serverless execution
_SERVERLESS_STATE = {
    "running": True,
    "paused": False,
    "step": 0,
    "score": 0,
    "fps": 10,
}


def app(environ, start_response):
    """WSGI application callable for Vercel Python runtime."""
    path = environ.get("PATH_INFO", "")
    method = environ.get("REQUEST_METHOD", "GET")

    # Common CORS headers
    headers = [
        ("Content-Type", "application/json"),
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type, Accept"),
    ]

    if method == "OPTIONS":
        start_response("200 OK", headers)
        return [b""]

    if method == "GET":
        if path.endswith("/status"):
            status_data = {
                "status": "ok",
                "mode": "vercel-serverless",
                "running": _SERVERLESS_STATE["running"],
                "paused": _SERVERLESS_STATE["paused"],
                "done": False,
                "step": _SERVERLESS_STATE["step"],
                "score": _SERVERLESS_STATE["score"],
                "fps": _SERVERLESS_STATE["fps"],
                "restrict_borders": False,
                "message": "Vercel serverless API ready. Browser client autonomous engine active.",
            }
            body = json.dumps(status_data).encode("utf-8")
            start_response("200 OK", headers)
            return [body]

        elif path.endswith("/stream"):
            stream_headers = [
                ("Content-Type", "text/event-stream"),
                ("Cache-Control", "no-cache"),
                ("Connection", "close"),
                ("Access-Control-Allow-Origin", "*"),
            ]
            init_payload = {
                "step": _SERVERLESS_STATE["step"],
                "score": _SERVERLESS_STATE["score"],
                "serverless": True,
                "message": "Vercel serverless connection established",
            }
            body = f"data: {json.dumps(init_payload)}\n\n".encode("utf-8")
            start_response("200 OK", stream_headers)
            return [body]

        else:
            body = json.dumps({"status": "ok", "path": path, "platform": "vercel"}).encode("utf-8")
            start_response("200 OK", headers)
            return [body]

    elif method == "POST":
        if not path.endswith("/control"):
            start_response("404 Not Found", headers)
            return [json.dumps({"status": "error", "message": f"Endpoint not found: {path}"}).encode("utf-8")]

        try:
            content_length = int(environ.get("CONTENT_LENGTH", 0) or 0)
            body_bytes = environ["wsgi.input"].read(content_length) if content_length > 0 else b"{}"
            body_json = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception as e:
            start_response("400 Bad Request", headers)
            return [json.dumps({"status": "error", "message": f"Invalid JSON body: {e}"}).encode("utf-8")]

        command = str(body_json.get("command", "")).strip().lower()

        if command == "play":
            _SERVERLESS_STATE["paused"] = False
            resp = {"status": "ok", "message": "Simulation running", "paused": False}
        elif command == "pause":
            _SERVERLESS_STATE["paused"] = True
            resp = {"status": "ok", "message": "Simulation paused", "paused": True}
        elif command == "step":
            _SERVERLESS_STATE["step"] += 1
            resp = {"status": "ok", "message": "Step executed", "step": _SERVERLESS_STATE["step"]}
        elif command == "reset":
            _SERVERLESS_STATE["step"] = 0
            _SERVERLESS_STATE["score"] = 0
            _SERVERLESS_STATE["paused"] = False
            resp = {"status": "ok", "message": "Simulation reset"}
        elif command == "set_speed":
            speed = body_json.get("value", body_json.get("speed", 10))
            try:
                _SERVERLESS_STATE["fps"] = max(1, min(60, int(speed)))
            except (ValueError, TypeError):
                pass
            resp = {"status": "ok", "fps": _SERVERLESS_STATE["fps"]}
        else:
            start_response("400 Bad Request", headers)
            return [json.dumps({"status": "error", "message": f"Unknown command: {command}"}).encode("utf-8")]

        start_response("200 OK", headers)
        return [json.dumps(resp).encode("utf-8")]

    start_response("405 Method Not Allowed", headers)
    return [b""]


# Expose both `app` and `handler` for WSGI compatibility
handler = app
