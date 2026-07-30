# ASymCat Architecture

This document defines the structure, boundaries, and design principles of
ASymCat. New modules should fit within it; changes that move away from it should
update it in the same PR.

---

## 1. Purpose and scope

ASymCat turns **co-occurring category pairs into measures of association**. Given
observations that pair a category from one variable with a category from another,
it computes how strongly the two are associated — and, crucially, in **each
direction independently** (X→Y and Y→X).

Asymmetry is the reason the library exists. Most association tooling reports a
single symmetric number (Cramér's V, mutual information); ASymCat keeps the two
directions apart, because "how well does the consonant predict the vowel" and
"how well does the vowel predict the consonant" are different questions with
different answers.

This is a **general-purpose** categorical-association tool, not a
computational-linguistics library. Linguistics is one important consumer, but the
core makes no assumption that categories are phonemes or words. Representative
domains:

| Domain | Variable X | Variable Y |
|--------|-----------|-----------|
| Linguistics | sound in language A | sound in language B (cognates) |
| Ecology | species | site / habitat |
| Genomics | allele | phenotype |
| Analytics | user segment | event category |
| ML features | categorical feature | class label |

**Design consequence:** the core vocabulary is *category*, *co-occurrence*,
*observation*, *pair* — never *word* or *phoneme*. Domain-specific convenience
(sequence readers, n-gram windows) is a thin ingestion layer, so an ecology or
analytics user never trips over linguistic terminology.

---

## 2. Design principles

1. **Directionality is first-class.** Every measure returns a value for *both*
   directions. A symmetric measure (Cramér's V, χ², mutual information) yields
   equal values; an asymmetric measure (Theil's U, Goodman–Kruskal λ, the
   Tresoldi measure) yields two different values. The return type never
   collapses the two directions into one.
2. **One scorer contract.** Build a `CatScorer` from co-occurrences, then call a
   named measure to score every pair. There is a single, predictable lifecycle,
   and each measure caches its result. Significance methods
   (`*_pvalue`, `bootstrap_ci`, `permutation_pvalue`) are *layers over* the same
   contract, not a parallel hierarchy.
3. **Robust probabilities via freqprob.** Smoothing (MLE, Laplace, Lidstone) is
   delegated to [`freqprob`](https://github.com/tresoldi/freqprob) rather than
   hand-rolled, so zero-count handling is principled and shared with the wider
   ecosystem. ASymCat treats freqprob estimators as log-probability sources and
   converts as needed.
4. **Significance is separable.** Point measures never change to accommodate
   significance. Parametric tests (χ², Fisher, log-likelihood ratio) and
   non-parametric resampling (permutation p-values, bootstrap confidence
   intervals) sit on top of the point estimates.
5. **KISS / DRY / YAGNI.** Prefer a small, obvious API over configurability
   nobody asked for. Shared preparation (contingency tables, co-occurrence
   collection) lives in exactly one place.
6. **Namespace-preserving refactors.** Internal reorganization must not break
   `import asymcat; asymcat.CatScorer(...)`. The public surface is defined by
   `asymcat/__init__.py`, independent of file layout.
7. **Typed and validated.** Type hints, `py.typed`, mypy; inputs are validated
   early with clear messages.

---

## 3. Package structure

`src/` layout; the public API is re-exported from `__init__.py`, so file layout
is transparent to users.

```
src/asymcat/
├── __init__.py       # public API surface (stable import path)
├── common.py         # data ingestion & preparation (readers, co-occurrence collection, contingency tables)
├── scorer.py         # CatScorer, the measures, parametric p-values, and resampling significance
├── correlation.py    # convenience wrappers over scorer for two aligned series
└── py.typed
```

**Why the layout is flat.** freqprob splits its estimators into `methods/` and
`performance/` subpackages because it hosts a large family of smoothing
algorithms and performance adapters. ASymCat's surface is small — data prep,
scoring, and a thin wrapper — so a taxonomy split would add indirection without
payoff (YAGNI). The three modules map cleanly onto the three concerns below, and
that is the right granularity until the codebase grows a genuine sub-family.

**Module responsibilities**

- **`common.py` — data in.** Readers (`read_sequences`, `read_pa_matrix`) and
  collectors (`collect_cooccs`, `collect_ngrams`, `collect_observations`,
  `collect_alphabets`) turn raw input into the list of category pairs the scorer
  consumes, plus `build_ct` for contingency tables. Depends only on the standard
  library, `numpy`, and `pandas`.
- **`scorer.py` — scoring.** `CatScorer` is the heart of the library. It holds
  the co-occurrences and exposes the measures — probabilistic (`mle`, `pmi`,
  `pmi_smoothed`), information-theoretic (`cond_entropy`, `theil_u`,
  `mutual_information`, `normalized_mutual_information`), statistical (`chi2`,
  `cramers_v`, `fisher`, `log_likelihood_ratio`), set-based (`jaccard_index`),
  predictive (`goodman_kruskal_lambda`), and the composite `tresoldi` measure —
  alongside parametric p-values (`chi2_pvalue`, `fisher_pvalue`,
  `log_likelihood_ratio_pvalue`) and resampling significance
  (`permutation_pvalue`, `bootstrap_ci`). Standalone `compute_*` helpers and
  `scale_scorer` support reuse and post-processing. Depends on `common`,
  `freqprob`, `numpy`, and `scipy`.
- **`correlation.py` — convenience.** Thin wrappers (`conditional_entropy`,
  `theil_u`, `cramers_v`) that take two parallel/aligned series and return a
  single summary statistic by delegating to `scorer`. This is the "I just have
  two columns" entry point.

### Dependency rule

Dependencies point **inward**: `correlation` imports from `scorer`; `scorer`
imports from `common`; `common` imports nothing else in the package. Nothing in
`common` reaches up into `scorer` or `correlation`.

---

## 4. Public API contract

The scorer lifecycle:

```python
import asymcat

# 1. Read and collect co-occurrences
data = asymcat.read_sequences("data.tsv")
cooccs = asymcat.collect_cooccs(data)

# 2. Build a scorer
scorer = asymcat.CatScorer(cooccs)

# 3. Call a measure -> {(x, y): (score_xy, score_yx)}
mle = scorer.mle()
theil = scorer.theil_u()  # asymmetric: the two directions differ

# 4. (optional) scale and assess significance
scaled = asymcat.scorer.scale_scorer(theil, method="minmax")
pvalues = scorer.permutation_pvalue("theil_u")
```

- Construction ingests the co-occurrences; each measure method scores **every
  observed pair at once** and returns a dict keyed by the `(x, y)` pair.
- Each dict value is a `(score_xy, score_yx)` tuple — the directional pair.
  Symmetric measures return equal components; asymmetric ones differ.
- Results are cached per measure, so repeated calls are cheap and significance
  methods can reuse the point estimates.

The public API is the top-level `asymcat` namespace: the readers and collectors
(`read_sequences`, `read_pa_matrix`, `collect_cooccs`, `collect_ngrams`,
`collect_observations`, `collect_alphabets`), `build_ct`, `CatScorer`, and the
`scorer` / `correlation` submodules. File layout under `src/asymcat/` is a
maintainer concern and may change without affecting these imports.

---

## 5. Terminology

To keep the library legible across domains, user-facing docs, examples, and
docstrings use neutral terms:

| Prefer | Avoid |
|--------|-------|
| category, symbol | word, phoneme (except in linguistic examples) |
| co-occurrence, pair | collocation |
| observation, series | corpus |
| variable X / variable Y | source / target language |

Domain examples (linguistics, ecology) are welcome in the guide and tutorials;
the shift is in the core API and prose, which stay domain-neutral.

---

## 6. Documentation architecture

- **Autogenerated API reference** from docstrings via `mkdocstrings` (no
  hand-maintained reference to drift) — see `docs/reference.md`.
- **Narrative guide** (hand-written): concepts, choosing a measure, worked
  examples — see `docs/USER_GUIDE.md`.
- **Custom landing page** via a MkDocs template override (`overrides/home.html`).
- **Tooling:** MkDocs-Material + mkdocstrings, published to GitHub Pages from CI
  (see `mkdocs.yml`; build with `make site` / `mkdocs serve`).

The earlier Sphinx site and rendered tutorial artifacts are retained under
`docs/archive/` and excluded from the built site. Generated artifacts (tutorial
HTML, figures, coverage reports) are kept out of version control.

---

## 7. Versioning & compatibility

The restructure to the `src/` layout, MkDocs documentation, and the shared
freqprob-style toolchain is **packaging- and tooling-only**: the public
`asymcat` namespace is unchanged, so existing user code keeps working without
edits. `MIGRATION.md` records what changed for users and contributors.

The public API is the top-level `asymcat` namespace. The project follows
semantic versioning; breaking changes to that surface are expected to be rare and
clearly flagged in the `CHANGELOG`.

---

## 8. Decisions (resolved)

1. **Flat module layout** kept over freqprob-style subpackages: the surface is
   small enough that `common` / `scorer` / `correlation` is the right
   granularity (§3, YAGNI).
2. **Smoothing delegated to freqprob** rather than reimplemented, so zero-count
   handling is principled and shared across the ecosystem (§2).
3. **Directional return type** (`{pair: (xy, yx)}`) is the single, uniform
   contract for every measure, symmetric or not (§2, §4).
4. **Significance layered, not baked in**: parametric and resampling tests wrap
   the point measures without altering them (§2).
5. **Docs tooling:** MkDocs-Material + mkdocstrings on GitHub Pages, replacing
   Sphinx (archived under `docs/archive/`) (§6).
