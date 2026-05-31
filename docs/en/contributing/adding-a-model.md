# Adding a New Model

## Step 1: Create Model Wrapper

Create `mlx_vc/models/<name>/__init__.py` and `model.py`:

```python
# mlx_vc/models/my_model/model.py
class MyModelVC:
    def __init__(self, verbose=True):
        self.sr = 22050
        self.sample_rate = self.sr

    def convert(self, source_audio, ref_audio, **kwargs):
        """Must return numpy array of converted audio."""
        from mlx_vc.backend import run_backend
        # ... delegate to backend
        return run_backend("my-model", source=..., reference=...)

    @property
    def model_info(self):
        return {"name": "MyModel", "type": "zero-shot", "sr": self.sr}
```

## Step 2: Create Backend Script

Create `mlx_vc/backends/my_model_infer.py`:

```python
#!/usr/bin/env python3
import argparse, json, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--args", type=str, required=True)
    args = json.loads(parser.parse_args().args)
    
    source = args["source"]
    reference = args["reference"]
    output = args["output"]
    
    # ... load model, run inference, save output
    
if __name__ == "__main__":
    main()
```

## Step 3: Register

In `mlx_vc/backend.py` add to `BACKENDS`:
```python
"my-model": {
    "script": "my_model_infer.py",
    "sample_rate": 22050,
    "description": "My model description",
},
```

In `mlx_vc/generate.py` add to `AVAILABLE_MODELS`:
```python
"my-model": {
    "class": "mlx_vc.models.my_model.MyModelVC",
    "description": "...",
    "default_repo": "...",
},
```

## Step 4: Test

Add tests in `mlx_vc/tests/` and run:
```bash
pytest -s mlx_vc/tests/ -v
```

## Step 5: Evaluate

Run the quality benchmark and record results in `BENCHMARK.md` Part B:

```bash
# Generate output
python -m mlx_vc.backend my-model --source src.wav --reference ref.wav --output out.wav

# Score it
python scripts/evaluate_quality.py \
    --source src.wav --reference ref.wav --outputs out.wav --json metrics.json
```

Add a new row to the appropriate table in `BENCHMARK.md` with the code version (`git describe --tags`) and metric values.

## Step 6: Document

Add `docs/models/my-model.md` and update `mkdocs.yml` nav.

## Step 7: Version & Changelog

Adding a new model is a **MINOR** bump. Update `CHANGELOG.md` under `[Unreleased]`:

```markdown
### Added
- **MyModel** backend: <one-line description> (BENCHMARK.md → v0.x.x Results)
```

Use commit type `feat`: `feat(models): add my-model backend`
