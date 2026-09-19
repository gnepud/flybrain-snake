"""Web Server & Real-time Neural Visualization Telemetry Engine.

Provides:
- SnakeServer: ThreadingHTTPServer serving static assets, SSE real-time neural telemetry,
  and simulation control endpoints.
- create_server: Factory function for creating a configured SnakeServer instance.
- run_server: Single-command entry point for starting the web server.
"""

import collections
import json
import mimetypes
import os
import queue
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from src.controller.flybrain_agent import FlyBrainAgent
from src.snake_env.env import SnakeEnv

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


class SnakeRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler routing static assets, SSE telemetry, and control APIs."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default HTTP access logging for clean test output."""
        return

    def _send_json(self, status_code: int, data: Dict[str, Any]) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, file_path: str, content_type: Optional[str] = None) -> None:
        """Sends static file with appropriate headers and caching disabled."""
        if not file_path.startswith(STATIC_DIR) or not os.path.isfile(file_path):
            self._send_json(404, {"status": "error", "message": "File not found"})
            return
        ext = os.path.splitext(file_path)[1].lower()
        ctype = content_type or MIME_TYPES.get(ext, mimetypes.guess_type(file_path)[0] or "application/octet-stream")
        with open(file_path, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self) -> None:
        """Handle CORS pre-flight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests for static pages, SSE stream, and status check."""
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            self._send_file(os.path.join(STATIC_DIR, "index.html"), "text/html; charset=utf-8")
        elif path == "/api/status":
            with self.server.lock:
                status_info = {
                    "status": "ok",
                    "running": not self.server.stopped,
                    "paused": self.server.paused,
                    "done": getattr(self.server.obs, "done", False),
                    "step": self.server.step_count,
                    "score": self.server.score,
                    "fps": self.server.fps,
                    "restrict_borders": getattr(self.server.env, "restrict_borders", False),
                }
            self._send_json(200, status_info)
        elif path == "/api/stream":
            self._handle_sse_stream()
        elif path == "/api/history":
            with self.server.lock:
                history_list = list(self.server.history)
            self._send_json(200, {"status": "ok", "history": history_list})
        else:
            rel_path = path.removeprefix("/static/").lstrip("/")
            candidate_path = os.path.normpath(os.path.join(STATIC_DIR, rel_path))
            if not candidate_path.startswith(STATIC_DIR):
                self._send_json(403, {"status": "error", "message": "Forbidden"})
            else:
                self._send_file(candidate_path)

    def _handle_sse_stream(self) -> None:
        """Streams real-time game telemetry and connectome spike activity."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        sub_queue: queue.Queue = queue.Queue(maxsize=100)
        with self.server.lock:
            self.server.subscribers.append(sub_queue)
            initial_frame = self.server.current_frame

        try:
            # Immediately emit the current frame upon connection
            if initial_frame is not None:
                payload = f"data: {json.dumps(initial_frame)}\n\n".encode("utf-8")
                self.wfile.write(payload)
                self.wfile.flush()

            while not self.server.stopped:
                try:
                    frame = sub_queue.get(timeout=0.5)
                    if frame is None:  # Sentinel unblock
                        break
                    payload = f"data: {json.dumps(frame)}\n\n".encode("utf-8")
                    self.wfile.write(payload)
                    self.wfile.flush()
                except queue.Empty:
                    # Heartbeat comment to preserve connection
                    try:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, socket.error, IOError):
                        self.close_connection = True
                        break
        except (BrokenPipeError, ConnectionResetError, socket.error, IOError):
            self.close_connection = True
        finally:
            self.close_connection = True
            with self.server.lock:
                if sub_queue in self.server.subscribers:
                    self.server.subscribers.remove(sub_queue)


    def do_POST(self) -> None:
        """Handle control commands (play, pause, step, reset, set_speed)."""
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path != "/api/control":
            self._send_json(404, {"status": "error", "message": f"Endpoint not found: {path}"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length < 0:
                raise ValueError("Negative Content-Length")
        except (ValueError, TypeError):
            self._send_json(400, {"status": "error", "message": "Invalid Content-Length header"})
            return

        try:
            body_bytes = self.rfile.read(content_length)
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception:
            self._send_json(400, {"status": "error", "message": "Malformed JSON payload"})
            return

        if not isinstance(body, dict):
            self._send_json(400, {"status": "error", "message": "JSON body must be an object"})
            return

        command = body.get("command")
        if not command or not isinstance(command, str):
            self._send_json(400, {"status": "error", "message": "Missing command parameter"})
            return

        command = command.strip().lower()

        if command == "play":
            frame = None
            with self.server.lock:
                if self.server.obs.done:
                    frame = self.server.reset_game()
                self.server.paused = False
            resp_data: Dict[str, Any] = {"status": "ok", "message": "Simulation running", "paused": False}
            if frame:
                resp_data["frame"] = frame
            self._send_json(200, resp_data)
        elif command == "pause":
            self.server.paused = True
            self._send_json(200, {"status": "ok", "message": "Simulation paused", "paused": True})
        elif command == "step":
            frame = self.server.step_simulation()
            self._send_json(200, {"status": "ok", "message": "Advanced 1 step", "frame": frame})
        elif command == "reset":
            seed_val = body.get("value")
            seed: Optional[int] = None
            if seed_val is not None:
                if isinstance(seed_val, bool) or isinstance(seed_val, float):
                    self._send_json(400, {"status": "error", "message": "Invalid seed value"})
                    return
                try:
                    seed = int(seed_val)
                except (ValueError, TypeError):
                    self._send_json(400, {"status": "error", "message": "Invalid seed value"})
                    return
            frame = self.server.reset_game(seed=seed)
            self._send_json(200, {"status": "ok", "message": "Simulation reset", "frame": frame})
        elif command == "set_speed":
            val = body.get("value")
            if val is None or isinstance(val, bool):
                self._send_json(400, {"status": "error", "message": "Invalid speed value"})
                return
            try:
                new_fps = float(val)
                if not (new_fps == new_fps) or abs(new_fps) == float("inf"):
                    self._send_json(400, {"status": "error", "message": "Invalid speed value"})
                    return
                self.server.fps = max(1.0, min(60.0, new_fps))
                self._send_json(200, {"status": "ok", "message": f"Speed updated to {self.server.fps} fps", "fps": self.server.fps})
            except (ValueError, TypeError):
                self._send_json(400, {"status": "error", "message": "Invalid speed value"})
        elif command in ("toggle_restrict_borders", "set_restrict_borders"):
            val = body.get("value")
            with self.server.lock:
                if val is not None:
                    new_state = bool(val)
                else:
                    new_state = not getattr(self.server.env, "restrict_borders", False)
                self.server.env.restrict_borders = new_state
                # Respawn food if currently violating inner zone restriction
                if new_state:
                    fx, fy = self.server.env._food
                    margin = 2
                    if (fx < margin or fx >= self.server.env.width - margin or
                        fy < margin or fy >= self.server.env.height - margin):
                        self.server.env._spawn_food()
                if self.server.current_frame:
                    self.server.current_frame["restrict_borders"] = new_state
                    self.server.current_frame["food"] = [int(x) for x in self.server.env._food]
                frame = self.server.current_frame or {}
            self._send_json(200, {
                "status": "ok",
                "message": f"Restrict borders set to {new_state}",
                "restrict_borders": new_state,
                "frame": frame
            })
        else:
            self._send_json(400, {"status": "error", "message": f"Unknown command: {command}"})


class SnakeServer(ThreadingHTTPServer):
    """Multi-threaded HTTP server hosting FlyBrain Snake simulation and SSE stream."""

    def __init__(
        self,
        server_address: Union[Tuple[str, int], str, None] = None,
        RequestHandlerClass: Any = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        fps: float = 10.0,
        grid_width: int = 16,
        grid_height: int = 16,
        seed: Optional[int] = None,
        restrict_borders: bool = False,
        **kwargs: Any,
    ):
        # Resolve address arguments
        if isinstance(server_address, tuple):
            addr = server_address
            handler = RequestHandlerClass or SnakeRequestHandler
        elif isinstance(server_address, str):
            if isinstance(RequestHandlerClass, int):
                addr = (server_address, RequestHandlerClass)
                handler = SnakeRequestHandler
            else:
                addr = (server_address, port)
                handler = RequestHandlerClass or SnakeRequestHandler
        else:
            addr = (host, port)
            handler = RequestHandlerClass or SnakeRequestHandler

        self.allow_reuse_address = True
        self.daemon_threads = True
        self._is_serving = False
        super().__init__(addr, handler)

        # Simulation state
        self.lock = threading.RLock()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.env = SnakeEnv(width=grid_width, height=grid_height, seed=seed, restrict_borders=restrict_borders)
        self.agent = FlyBrainAgent()
        self.paused = False
        self.stopped = False
        self.fps = float(fps)
        self.step_count = 0
        self.score = 0
        self.current_frame: Optional[Dict[str, Any]] = None
        self.history: collections.deque = collections.deque(maxlen=1000)
        self.subscribers: List[queue.Queue] = []

        # Synchronization & background worker
        self._stop_event = threading.Event()
        self._step_once_event = threading.Event()

        # Initialize simulation
        self.reset_game(seed=seed)

        # Launch background loop
        self.sim_thread = threading.Thread(target=self._simulation_loop, name="SnakeSimLoop", daemon=True)
        self.sim_thread.start()

    def _build_frame(
        self,
        obs: SnakeEnv,
        action: str = "NONE",
        epg: Optional[List[float]] = None,
        steering: Optional[Dict[str, float]] = None,
        motor: Optional[Dict[str, float]] = None,
        ate_food: bool = False,
        spikes: Optional[List[int]] = None,
        spikes_count: int = 0,
        olfactory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "step": self.step_count,
            "score": self.score,
            "head": [int(x) for x in obs.head],
            "body": [[int(x), int(y)] for x, y in obs.body],
            "food": [int(x) for x in obs.food],
            "direction": int(obs.direction),
            "done": bool(obs.done),
            "reason": obs.reason,
            "epg": epg or [0.0] * 16,
            "steering": steering or {"pfl3_l": 0.0, "pfl3_r": 0.0, "threshold": 0.03, "is_override": False},
            "motor": motor or {"dna02_l": 0.0, "dna02_r": 0.0, "dna01": 0.0, "dnp": 0.0, "is_override": False},
            "action": action,
            "restrict_borders": bool(getattr(self.env, "restrict_borders", False)),
            "sensory": {
                "food_bearing": float(getattr(obs, "food_bearing", 0.0)),
                "dist_front": int(getattr(obs, "dist_front", 10)),
                "dist_left": int(getattr(obs, "dist_left", 10)),
                "dist_right": int(getattr(obs, "dist_right", 10)),
            },
            "ate_food": ate_food,
            "spikes": spikes or [],
            "spikes_count": spikes_count,
            "olfactory": olfactory or {"c_left": 0.0, "c_right": 0.0, "diff": 0.0},
        }

    def reset_game(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Resets the game environment and connectome agent dynamics."""
        with self.lock:
            self.obs = self.env.reset(seed=seed)
            self.agent.reset()
            self.step_count = int(self.obs.step_count)
            self.score = int(self.obs.score)

            frame = self._build_frame(self.obs)
            self.current_frame = frame
            self.history.append(frame)
            self._broadcast_frame(frame)
            return frame

    def step_simulation(self) -> Dict[str, Any]:
        """Advances the simulation by one discrete step."""
        with self.lock:
            if self.obs.done:
                return self.current_frame or {}

            action, info = self.agent.act(self.obs)
            next_obs, reward, _, _ = self.env.step(action)
            self.obs = next_obs
            self.step_count = int(next_obs.step_count)
            self.score = int(next_obs.score)

            action_name = action.name if hasattr(action, "name") else str(action)
            steering = info.get("steering", {})
            is_override = bool(steering.get("is_override", False))
            forward_val = float(info.get("dna01", info.get("dnp", 0.0)))
            motor = {
                "dna02_l": float(info.get("dna02_l", 0.0)),
                "dna02_r": float(info.get("dna02_r", 0.0)),
                "dna01": forward_val,
                "dnp": forward_val,
                "is_override": is_override,
            }
            epg = [float(x) for x in info.get("epg", [0.0] * 16)]
            spikes = info.get("spikes", [])

            frame = self._build_frame(
                next_obs,
                action=action_name,
                epg=epg,
                steering={
                    "pfl3_l": float(steering.get("pfl3_l", 0.0)),
                    "pfl3_r": float(steering.get("pfl3_r", 0.0)),
                    "threshold": float(steering.get("threshold", 0.03)),
                    "is_override": is_override,
                },
                motor=motor,
                ate_food=bool(reward > 0),
                spikes=spikes,
                spikes_count=info.get("spikes_count", len(spikes)),
                olfactory=info.get("olfactory"),
            )
            self.current_frame = frame
            self.history.append(frame)
            self._broadcast_frame(frame)
            return frame

    def _broadcast_frame(self, frame: Dict[str, Any]) -> None:
        """Pushes a telemetry frame to all connected SSE clients."""
        for sub in list(self.subscribers):
            try:
                sub.put_nowait(frame)
            except queue.Full:
                pass

    def _simulation_loop(self) -> None:
        """Background thread executing the continuous or stepped game loop."""
        while not self._stop_event.is_set():
            if self.paused:
                if self._step_once_event.is_set():
                    self._step_once_event.clear()
                    self.step_simulation()
                else:
                    time.sleep(0.02)
                continue

            # Preserve death state: pause on game over and wait for manual reset/play
            if self.obs.done:
                with self.lock:
                    self.paused = True
                time.sleep(0.02)
                continue

            self.step_simulation()
            fps = max(1.0, min(60.0, float(self.fps)))
            time.sleep(1.0 / fps)

    def start(self) -> None:
        """Alias for serve_forever."""
        self.serve_forever()

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        """Handle requests until shutdown is requested."""
        self._is_serving = True
        try:
            super().serve_forever(poll_interval=poll_interval)
        finally:
            self._is_serving = False

    def shutdown(self) -> None:
        """Signals background threads and shuts down HTTP server."""
        self.stopped = True
        self._stop_event.set()
        # Unblock subscribers
        with self.lock:
            for sub in list(self.subscribers):
                try:
                    sub.put_nowait(None)
                except Exception:
                    pass
        if getattr(self, "_is_serving", False):
            super().shutdown()

    def stop(self) -> None:
        """Stops the server, closes sockets, and terminates worker threads."""
        self.shutdown()
        try:
            self.server_close()
        except Exception:
            pass
        if hasattr(self, "sim_thread") and self.sim_thread.is_alive():
            self.sim_thread.join(timeout=0.5)


def create_server(host: str = "127.0.0.1", port: int = 8080, **kwargs: Any) -> SnakeServer:
    """Factory creating an instance of SnakeServer."""
    return SnakeServer(host=host, port=port, **kwargs)


def run_server(host: str = "0.0.0.0", port: int = 8080, **kwargs: Any) -> None:
    """Single-command runner starting the HTTP & SSE server."""
    server = create_server(host=host, port=port, **kwargs)
    print(f"Starting FlyBrain Snake Server at http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.stop()
