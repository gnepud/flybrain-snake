#!/usr/bin/env python3
"""Single-command server launcher for FlyBrain Snake Web UI."""

import argparse
import sys
from src.web.server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlyBrain Snake: Connectome Sensorimotor Neural Circuit Web Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host interface to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Simulation steps per second (default: 10.0)",
    )
    args = parser.parse_args()
    try:
        run_server(host=args.host, port=args.port, fps=args.fps)
    except KeyboardInterrupt:
        print("\nFlyBrain Snake server stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
