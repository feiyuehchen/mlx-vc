# Development Setup

## Clone and Install

```bash
git clone https://github.com/feiyuehchen/mlx-vc.git
cd mlx-vc
uv venv && source .venv/bin/activate
uv pip install -e ".[all,dev]"
```

Requires Python 3.10+ (pinned in `.python-version`).

## Git Hooks

Install both pre-commit (formatting) and commit-msg (conventional commits) hooks:

```bash
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

This enforces:

- **Black** (line-length 88) + **isort** (black profile) on staged files
- **Commitizen** validates commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) format

## Run Tests

```bash
pytest -s mlx_vc/tests/ -v
```

## Code Style

```bash
pre-commit run --all-files  # Run formatters manually
```

## Commit Messages

```
<type>[scope]: <description>
```

Types: `feat`, `fix`, `perf`, `bench`, `docs`, `test`, `refactor`, `ci`, `chore`, `style`

Add `!` for breaking changes: `feat!: remove deprecated endpoint`

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for version bump rules.

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

## Key Files

| File | Role |
|------|------|
| `BENCHMARK.md` | Quality metric definitions (Part A) + accumulated results (Part B) |
| `CHANGELOG.md` | User-visible change history per version |
| `CLAUDE.md` | AI agent guidance + architecture notes |
| `.python-version` | Python version pin (3.10) |
