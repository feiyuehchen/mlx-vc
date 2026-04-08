# Development Setup

## Clone and Install

```bash
git clone https://github.com/feiyuehchen/mlx-vc.git
cd mlx-vc
uv venv && source .venv/bin/activate
uv pip install -e ".[all,dev]"
```

## Run Tests

```bash
pytest -s mlx_vc/tests/ -v
```

## Code Style

We use Black (line-length=88) and isort (black profile):

```bash
pre-commit install          # Set up git hooks
pre-commit run --all-files  # Run manually
```

## Project Structure

```
mlx_vc/
├── models/        # Model wrappers (unified API)
├── backends/      # Subprocess inference scripts
├── demo/          # Real-time demo
├── tests/         # Test suite
├── server.py      # FastAPI server
├── backend.py     # Subprocess runner
├── generate.py    # CLI entry point
└── audio_io.py    # Audio utilities
```
