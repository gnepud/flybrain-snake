# FlyBrain Snake

Autonomous Snake game guided by Drosophila whole-brain connectome dynamics (`flybrain`), featuring a real-time web visualization.

## Quick Start

```bash
pip install -r requirements.txt
python3 run_server.py
```

Open [http://localhost:8080](http://localhost:8080).

## Tests & Benchmark

```bash
pytest
python3 -m src.benchmark.runner --episodes 50
```
