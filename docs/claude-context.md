## Versioning — 4 locations must stay in sync

When bumping the package version, **all four locations must be updated together** or the PyPI release will be broken:

| File | Field |
|------|-------|
| `pyproject.toml` | `version = "X.Y.Z"` under `[project]` |
| `rust/arithma_core/Cargo.toml` | `version = "X.Y.Z"` under `[package]` |
| `python/arithma/__init__.py` | `__version__ = "X.Y.Z"` |
| Git tag | `vX.Y.Z` pushed to origin |

After committing and tagging, create a **GitHub Release** from the tag to trigger the PyPI publish workflow (`release: published` event). The workflow does **not** fire on a tag push alone.

> **Important:** `pyproject.toml version` is what PyPI indexes — not the git tag. Pre-release suffixes like `2.0.0a0` are **not** matched by `>=2.0.0` in pip. Always use clean `X.Y.Z` for stable releases.

# Arithma

General engineering standards: ~/.claude/docs/standards.md.

### Project Structure

```
arithma/
├── pyproject.toml
├── rust/arithma_core/        # Rust crate (cdylib + lib)
├── python/arithma/           # Python package (maturin python-source)
│   ├── __init__.py           # public surface + _HAS_RUST guard
│   └── _launcher.py          # arithma-app launcher
└── tests/
```
