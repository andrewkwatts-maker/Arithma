# Publishing Arithma

Two registries, one source tree: `arithma_core` to crates.io and `arithma` to
PyPI. Both project names are already owned.

## Open blocker — the licence contradicts itself

This is the one item that cannot be closed by anyone reading the code, and it
has to be closed **before** either upload.

`LICENSE` is the MIT licence. `rust/arithma_core/Cargo.toml` declares
`license = "MIT"`. `pyproject.toml` declares `license = { file = "LICENSE" }`
and the `License :: OSI Approved :: MIT License` classifier.

And **62 source files** open with:

```
copyright (c) 2025 Andrew Keith Watts. All rights reserved.

This is the intellectual property of Andrew Keith Watts. Unauthorized
reproduction, distribution, or modification of this code, in whole or in part,
without the express written permission of Andrew Keith Watts is strictly
prohibited.
```

Those two statements cannot both be the terms. MIT grants exactly the rights the
header prohibits. The banner is the PlayTow datasheet convention carried over
from `pt-arithmos` — `constants.rs` still names it as such where it strips the
same banner out of shipped JSON — so the likeliest reading is that it predates
the decision to open-source. But *likeliest* is not a basis for choosing licence
terms on someone's behalf, and publishing under MIT with a proprietary notice in
every file is worse than publishing under either one cleanly.

Resolve it one of two ways.

**If MIT is intended** — strip the banner from every file:

```bash
# Dry run first; it rewrites 62 files.
python - <<'EOF'
import io, pathlib, re
BANNER = re.compile(
    r"^//====== .*? ======//\n(?://!.*\n)*?//!For inquiries[^\n]*\n\n?",
    re.MULTILINE,
)
for p in pathlib.Path("rust/arithma_core/src").rglob("*.rs"):
    t = io.open(p, encoding="utf-8").read()
    n = BANNER.sub("", t, count=1)
    if n != t:
        print("would strip:", p)
EOF
```

Replace the banner with a one-line SPDX marker (`// SPDX-License-Identifier:
MIT`) if a per-file notice is wanted at all.

**If proprietary is intended** — then neither registry upload should happen, and
`Cargo.toml`, `pyproject.toml` and `LICENSE` all need correcting to match. Note
that the four vendored dependants consume Arithma as a git submodule, so they
are unaffected either way.

## Everything else is ready

| Item | State |
|---|---|
| crates.io metadata | `repository`, `homepage`, `documentation`, `readme`, `keywords`, `categories`, `rust-version`, `exclude` — all set |
| PyPI metadata | `readme`, `keywords`, 8 classifiers, `[project.urls]` — all set |
| `cargo package` | passes; 68 files, 277 KiB compressed |
| Version | `2.0.4` in `pyproject.toml`, `Cargo.toml` and `arithma.__version__` |
| MSRV | `1.82`, driven by `Option::is_none_or` at `pyfacade/geometry.rs:143` and `unit.rs:473` |
| Tests | 645 with the `python` feature, 505 without, 136 pytest |
| Lints | `cargo fmt --check` and `clippy -D warnings` clean on every feature combination |
| README | rewritten; it described eight implemented areas as `unimplemented!()` panics |

### The MSRV is not a guess and should not be relaxed casually

`rust-version = "1.82"` is driven by `Option::is_none_or`, which is used at
exactly two sites and **only inside `debug_assert!`**. Both compile in every
build regardless, so the requirement is real. Reading the MSRV off the newest
*language* feature instead (`let ... else`, 1.65) would have declared 1.74 and
broken the build for anyone who believed it. If the floor needs to come down,
rewrite those two assertions as `map_or` and verify with `cargo-msrv` rather
than by inspection.

## Release order

1. Close the licence question above.
2. `cargo publish -p arithma_core --dry-run`, then for real. crates.io is
   **irreversible** — a published version can be yanked but never replaced, so
   the dry run is not optional.
3. Tag `v2.0.4`. CI builds wheels for Linux, Windows and macOS (x86_64 and
   aarch64) and publishes to PyPI through the `pypi` environment via trusted
   publishing.
4. Only then remove the `Arithmos*` aliases — see [RENAME.md](RENAME.md). They
   are the migration mechanism for the four vendored dependants and must not
   drop before those repos have moved.
