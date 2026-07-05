# CBBIO Versioning and Branching Guide

This document defines how versions are numbered, when to bump them, what each version number means, and how branches are used. Follow it consistently so that users and contributors can reason about what a version number promises.

---

## Version Number Meaning

CBBIO uses **Semantic Versioning** (SemVer): `MAJOR.MINOR.PATCH`.

```
0.4.2
│ │ └── PATCH  — bug fix, no API change
│ └──── MINOR  — new feature, backwards compatible
└────── MAJOR  — breaking change
```

### MAJOR (`1.0.0`, `2.0.0`, …)

Increment when you make a **breaking change** — something that requires a user to change their code.

Breaking changes include:
- Removing a public function, class, or parameter
- Renaming a public function, class, or parameter
- Changing the return type or shape of a public function in an incompatible way
- Changing required parameters or their order
- Dropping support for a previously supported Python version

> The project is currently pre-1.0 (`0.x.y`). During pre-1.0, minor bumps (`0.2.0`, `0.3.0`) carry the same weight as a major release: they may contain breaking changes. A `0.x` version makes no stability promise.

### MINOR (`0.2.0`, `0.3.0`, …)

Increment when you add **new public API** that is backwards compatible.

New features include:
- New public functions, classes, or methods
- New optional parameters added at the end of an existing signature
- New model families added to the embedding generator registry
- New datasets added to the probing catalog
- Substantial new capabilities (new search backend, new probing objective, new output format)

### PATCH (`0.1.1`, `0.1.2`, …)

Increment when you fix a bug **without adding new API or breaking existing behaviour**.

Patch changes include:
- Bug fixes in existing functions
- Performance improvements that do not change the API
- Documentation corrections
- Dependency pin updates that do not change runtime behaviour

---

## Pre-release Suffixes

Use pre-release suffixes for versions that are not ready for general use.

| Suffix | When to use | Example |
|---|---|---|
| `aN` | Alpha — incomplete, may change significantly | `0.2.0a0` |
| `bN` | Beta — feature-complete but potentially buggy | `0.2.0b0` |
| `rcN` | Release candidate — expected to ship as-is | `0.2.0rc0` |

Start the counter at `0`. Increment only within the same version (`b0` → `b1`, not `b0` → `rc0` without a reason).

The current project is in beta (`0.1b1`). Do not use alpha suffixes unless you are prototyping something that is definitely not ready for others to depend on.

---

## When to Bump the Version

| What you did | Version change |
|---|---|
| Fixed a bug, no API change | `PATCH` bump |
| Added a new public function / class / parameter | `MINOR` bump |
| Added a new embedding model family | `MINOR` bump |
| Added a new probing dataset or catalog entry | `MINOR` bump |
| Removed or renamed a public name | `MAJOR` bump (or `MINOR` during `0.x`) |
| Changed the return type of a public function | `MAJOR` bump (or `MINOR` during `0.x`) |
| Updated docs only | No bump |
| Updated tests only | No bump |
| Reformatted code (no logic change) | No bump |
| Updated a dependency pin (no behaviour change) | No bump |

When in doubt, err on the side of bumping `MINOR`. Bumping too often costs nothing; not bumping when you should breaks users silently.

---

## Where the Version Lives

The version is declared in exactly one place: `pyproject.toml`.

```toml
[tool.poetry]
version = "0.1.0"
```

Do not hardcode the version anywhere else. If code needs the version at runtime, read it from the package metadata:

```python
from importlib.metadata import version
__version__ = version("biodata")
```

---

## Branching Model

### Long-lived branches

| Branch | Purpose |
|---|---|
| `main` | Always releasable. Every commit on `main` could be tagged and shipped. |

There is no separate `develop` branch. All integration happens on `main` via pull requests from feature branches.

### Feature branches

Create a feature branch for every non-trivial change. Branch from `main`.

**Naming:**

```
feature/<short-description>
fix/<short-description>
chore/<short-description>
docs/<short-description>
```

- `feature/` — new functionality
- `fix/` — bug fix
- `chore/` — dependency updates, refactoring, tooling changes
- `docs/` — documentation only changes

Examples:
```
feature/amplify-embedding-backend
fix/esm1b-hf-weight-mismatch
chore/update-torch-pin
docs/probing-guide
```

Keep branch names lowercase and hyphenated. No ticket numbers, no dates.

### Branch lifetime

Delete branches as soon as they are merged. Long-lived feature branches accumulate drift and cause painful rebases.

---

## Commit Messages

Every commit message has a short subject line (≤ 72 characters) and optionally a body.

**Subject line format:**

```
<type>: <imperative short description>
```

Types match the branch prefix:

| Type | Use when |
|---|---|
| `feat` | Adding new functionality |
| `fix` | Fixing a bug |
| `chore` | Refactoring, tooling, dependency updates |
| `docs` | Documentation only |
| `test` | Test changes only |
| `perf` | Performance improvement without API change |

Examples:
```
feat: Add AMPLIFY embedding backend
fix: Correct ESM-1b pre/post-norm weight loading via torch.hub
docs: Add probing subsystem guide
chore: Update torch pin to 2.11.0
test: Add residue probe evaluation edge cases
```

The subject line must use the imperative mood ("Add", "Fix", "Update") not past tense ("Added", "Fixed"). It must not end with a period.

If the commit body is needed, leave one blank line after the subject and explain **why** the change was made (not what — the diff shows that):

```
fix: Correct ESM-1b pre/post-norm weight loading via torch.hub

The HuggingFace facebook/esm-1b checkpoint was trained with post-norm
but the HF model code uses pre-norm. Loading via torch.hub avoids the
mismatch. The accuracy loss was ~10% on secondary structure prediction.
```

---

## Release Process

### Step 1 — decide the new version

Apply the versioning rules above. If any merged feature since the last release changes the public API, the new version is at least a MINOR bump.

### Step 2 — update `pyproject.toml`

```toml
version = "0.2.0"
```

### Step 3 — commit the version bump

```bash
git add pyproject.toml
git commit -m "chore: Bump version to 0.2.0"
```

The commit message for a version bump is always `chore: Bump version to X.Y.Z`.

### Step 4 — tag the release

```bash
git tag -a 0.2.0 -m "Release 0.2.0"
```

Use annotated tags (`-a`), not lightweight tags. The tag name is the bare version number (no `v` prefix).

### Step 5 — push

```bash
git push origin main --tags
```

---

## Release Checklist

Before tagging a release:

- [ ] All tests pass: `poetry run pytest -q`
- [ ] Type checking passes: `poetry run pyright`
- [ ] Version in `pyproject.toml` is correct and follows the rules above
- [ ] `pyproject.toml` version is the only place the version is declared
- [ ] All breaking changes are accounted for in the version bump
- [ ] The commit is on `main` (not a feature branch)
- [ ] The tag is annotated: `git tag -a X.Y.Z -m "Release X.Y.Z"`
- [ ] No uncommitted changes: `git status` is clean

---

## Removal Policy

Before removing or renaming a public name, account for it in the version bump and update the
documentation in the same change. CBBIO is pre-1.0, so removals land in the next MINOR release.
Do not add compatibility aliases for removed names.

Document the deprecation in the relevant doc file and in the commit message.
