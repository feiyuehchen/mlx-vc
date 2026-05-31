# 新增模型

## Step 1: 建立 Model Wrapper

建立 `mlx_vc/models/<name>/__init__.py` 和 `model.py`：

```python
# mlx_vc/models/my_model/model.py
class MyModelVC:
    def __init__(self, verbose=True):
        self.sr = 22050
        self.sample_rate = self.sr

    def convert(self, source_audio, ref_audio, **kwargs):
        """Must return numpy array of converted audio."""
        from mlx_vc.backend import run_backend
        return run_backend("my-model", source=..., reference=...)

    @property
    def model_info(self):
        return {"name": "MyModel", "type": "zero-shot", "sr": self.sr}
```

## Step 2: 建立 Backend Script

建立 `mlx_vc/backends/my_model_infer.py`：

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

## Step 3: 註冊

在 `mlx_vc/backend.py` 的 `BACKENDS` 加入：
```python
"my-model": {
    "script": "my_model_infer.py",
    "sample_rate": 22050,
    "description": "My model description",
},
```

在 `mlx_vc/generate.py` 的 `AVAILABLE_MODELS` 加入：
```python
"my-model": {
    "class": "mlx_vc.models.my_model.MyModelVC",
    "description": "...",
    "default_repo": "...",
},
```

## Step 4: 測試

在 `mlx_vc/tests/` 加測試並執行：
```bash
pytest -s mlx_vc/tests/ -v
```

## Step 5: 評估

跑品質 benchmark 並把結果記錄到 `BENCHMARK.md` Part B：

```bash
# 產生輸出
python -m mlx_vc.backend my-model --source src.wav --reference ref.wav --output out.wav

# 評分
python scripts/evaluate_quality.py \
    --source src.wav --reference ref.wav --outputs out.wav --json metrics.json
```

在 `BENCHMARK.md` 對應表格新增一行，包含 code version（`git describe --tags`）和指標值。

## Step 6: 文件

新增 `docs/models/my-model.md` 並更新 `mkdocs.yml` nav。

## Step 7: 版號與 Changelog

新增模型 = **MINOR** bump。更新 `CHANGELOG.md` 的 `[Unreleased]`：

```markdown
### Added
- **MyModel** backend: <一行描述> (BENCHMARK.md → v0.x.x Results)
```

Commit type 用 `feat`：`feat(models): add my-model backend`
