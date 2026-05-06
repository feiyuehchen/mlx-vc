# Contributing to mlx-vc

Thanks for your interest in mlx-vc!  This guide covers everything you need to set up, contribute, and ship.

## Quick start

```bash
git clone git@github.com:feiyuehchen/mlx-vc.git
cd mlx-vc
uv venv && source .venv/bin/activate
uv pip install -e ".[all,dev]"
pre-commit install
```

Run the tests:

```bash
pytest -s mlx_vc/tests/ -v --timeout=60
```

Most backends additionally require cloning a sibling reference repo (e.g. `../seed-vc-ref/`).  Each backend's docstring lists the exact setup steps.

## Project layout

```
mlx_vc/
├── backend.py        # BACKENDS registry + subprocess runner
├── jobs.py           # In-memory JobManager
├── server.py         # FastAPI: convert / batch / WS realtime
├── realtime.py       # OpenVoiceSession singleton
├── generate.py       # CLI + AVAILABLE_MODELS in-process registry
├── audio_io.py       # WAV load/save
├── models/           # In-process Python wrappers
└── backends/         # Subprocess inference scripts
```

See `README.md` for a detailed architecture overview and `CLAUDE.md` for the device matrix per backend.

## Development workflow

1. **Create a branch**: `git switch -c feat/my-thing`
2. **Make changes**: prefer editing existing files; add new ones only when necessary.
3. **Run pre-commit**: `pre-commit run --all-files` (auto-runs on `git commit` once `pre-commit install` is done)
4. **Run tests**: `pytest -s mlx_vc/tests/ -v --timeout=60`
5. **Commit**: messages follow `<type>: <summary>` (`feat:`, `fix:`, `perf:`, `docs:`, `test:`, `style:`, `ci:`, `refactor:`)
6. **Open a PR**: fill in the PR template; CI must pass before review.

## Code style

- **black** (line-length 88, target py310) — handled by pre-commit
- **isort** (black profile) — handled by pre-commit
- **type hints** are encouraged on public APIs.  Internal helper functions can skip them.
- **docstrings** are encouraged on classes and public functions.  Use one-line summary + optional `Args:`/`Returns:` sections.

Avoid:
- Adding error handling for cases that can't happen (trust framework guarantees)
- Comments that explain *what* code does (the code shows that); only comment *why*
- Backward-compatibility shims unless explicitly requested

## Adding a new VC model

The fast path uses subprocess isolation:

1. Create `mlx_vc/backends/<name>_infer.py`.  Read `--args` JSON, run inference, write `output`.  Look at any existing backend (e.g. `seed_vc_infer.py`) as a template.
2. Register it in `BACKENDS` (`mlx_vc/backend.py`):
   ```python
   "<name>": {
       "script": "<name>_infer.py",
       "sample_rate": <sr>,
       "description": "<short description>",
       # optional:
       # "extra_args": {"variant": "...", "max_ref_seconds": 10.0},
   }
   ```
3. Add a setup-recipe section to your script's docstring (clone steps, weight URLs, dep installs).
4. Verify it runs end-to-end on at least one source/reference pair before opening the PR.

If you also want a Python class wrapper (in-process API), add it under `mlx_vc/models/<name>/` and register in `AVAILABLE_MODELS` (`mlx_vc/generate.py`).

## Tests

- New unit tests go in `mlx_vc/tests/`.
- Tests should be **fast** (default 60s timeout) and **deterministic**.
- Tests that depend on `[server]` extras must `pytest.importorskip("fastapi")`.
- Don't import heavy model deps at test-collection time — use lazy fixtures.

## Reporting issues

Use the [issue templates](.github/ISSUE_TEMPLATE) for bugs, feature requests, and model integration requests.  When reporting a bug:

- Include the exact `python -V` output, `pip show torch torchaudio | head` output, and macOS version
- Quote the full traceback (truncated stderr is hard to debug)
- Mention which backend (`seed-vc`, `meanvc`, etc.) the bug applies to

## Security issues

See [`SECURITY.md`](SECURITY.md).  Do **not** open public issues for security-sensitive bugs.

## License

By contributing, you agree your contributions will be licensed under the MIT License (see [`LICENSE`](LICENSE)).
