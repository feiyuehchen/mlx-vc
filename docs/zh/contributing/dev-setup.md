# 開發環境設置

## Clone 與安裝

```bash
git clone https://github.com/feiyuehchen/mlx-vc.git
cd mlx-vc
uv venv && source .venv/bin/activate
uv pip install -e ".[all,dev]"
```

需要 Python 3.10+（`.python-version` 已固定）。

## Git Hooks

安裝 pre-commit（格式化）和 commit-msg（Conventional Commits）hooks：

```bash
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

這會強制執行：

- **Black**（行寬 88）+ **isort**（black profile）格式化
- **Commitizen** 驗證 commit message 符合 [Conventional Commits](https://www.conventionalcommits.org/) 格式

## 跑測試

```bash
pytest -s mlx_vc/tests/ -v
```

## Code Style

```bash
pre-commit run --all-files  # 手動跑格式化
```

## Commit Messages

```
<type>[scope]: <description>
```

Types: `feat`, `fix`, `perf`, `bench`, `docs`, `test`, `refactor`, `ci`, `chore`, `style`

Breaking changes 加 `!`：`feat!: remove deprecated endpoint`

版號影響規則見 [CONTRIBUTING.md](../../../CONTRIBUTING.md)。

## 專案結構

```
mlx_vc/
├── models/        # Model wrappers（統一 API）
├── backends/      # Subprocess 推論腳本
├── demo/          # 即時 demo
├── tests/         # 測試
├── server.py      # FastAPI server
├── backend.py     # Subprocess runner
├── generate.py    # CLI 入口
└── audio_io.py    # Audio 工具
```

## 關鍵文件

| 文件 | 角色 |
|------|------|
| `BENCHMARK.md` | 品質指標定義（Part A）＋累積結果（Part B） |
| `CHANGELOG.md` | 每版使用者可見的變更歷史 |
| `CLAUDE.md` | AI agent 指引 + 架構說明 |
| `.python-version` | Python 版本固定（3.10） |
