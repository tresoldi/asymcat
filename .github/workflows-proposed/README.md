# Proposed workflows

These three files are the freqprob-aligned replacements for asymcat's CI, staged
here because they could not be committed under `.github/workflows/` by the
automation that authored them (updating files under `.github/workflows/`
requires the `workflow` OAuth scope). Move them into place and commit locally:

```bash
git mv .github/workflows-proposed/docs.yml    .github/workflows/docs.yml
git mv .github/workflows-proposed/quality.yml .github/workflows/quality.yml
git mv .github/workflows-proposed/release.yml .github/workflows/release.yml
rmdir .github/workflows-proposed 2>/dev/null || rm -f .github/workflows-proposed/README.md
git commit -am "Align CI workflows with the freqprob structure"
```

## What each file does

- **docs.yml** *(new)* — builds the MkDocs site with `mkdocs build --strict` and
  deploys it to GitHub Pages. Without it the migrated docs site never publishes.
  Requires Pages to be enabled for the repo (Settings → Pages → Source: GitHub
  Actions).
- **quality.yml** *(replaces the existing one)* — splits into a `quality` job
  (ruff format check, ruff lint, mypy, bandit on `src/`) and a `test` job with an
  OS matrix (Ubuntu/macOS/Windows × Python 3.10–3.12) uploading coverage to
  Codecov. Fixes the stale `mypy asymcat/ tests/` path from before the `src/`
  migration.
- **release.yml** *(replaces the existing one)* — minimal `build → publish`
  pipeline using PyPI trusted publishing (OIDC), triggered on `v*` tags.

## Repository settings these assume

- **Codecov:** set the `CODECOV_TOKEN` repository secret (quality.yml).
- **PyPI trusted publishing:** register this repo/workflow as a trusted
  publisher on PyPI, and add a `pypi` environment (release.yml). No API token
  needed.
- **Pages:** Settings → Pages → Source: "GitHub Actions" (docs.yml).
