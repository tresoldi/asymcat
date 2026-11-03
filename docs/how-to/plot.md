# Plot score matrices

A heatmap shows the scores of all pairs at once. `scorer2matrices` arranges a
measure as two matrices, one per direction, which any plotting library can
draw. The examples use matplotlib, which `pip install "asymcat[viz]"`
installs.

## From scores to matrices

```pycon
>>> import asymcat
>>> from asymcat.scorer import scorer2matrices
>>> words = [
...     ("cat", "kat"),
...     ("cot", "kot"),
...     ("city", "siti"),
...     ("kit", "kit"),
...     ("cent", "sent"),
...     ("cell", "sel-"),
...     ("kid", "kid"),
...     ("coin", "koin"),
... ]
>>> scorer = asymcat.CatScorer(asymcat.collect_cooccs(words, order=1))
>>> xy, yx, letters, sounds = scorer2matrices(scorer.mle())
>>> xy.shape, yx.shape
((11, 11), (11, 11))
```

`xy[i, j]` is the score of `letters[i]` as a predictor of `sounds[j]`, here
$P(\text{sound} \mid \text{letter})$; `yx[j, i]` is the score of `sounds[j]` as
a predictor of `letters[i]`. The alphabets are sorted, and the gap symbol `-`
marks the silent second `l` of "cell".

## A pair of heatmaps

Drawing the two directions side by side, with the same colour scale, shows
where prediction runs one way:

```pycon
>>> import matplotlib
>>> matplotlib.use("Agg")
>>> import matplotlib.pyplot as plt
>>> fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
>>> panels = [(xy, letters, sounds, "P(sound | letter)"), (yx, sounds, letters, "P(letter | sound)")]
>>> for ax, (matrix, rows, cols, title) in zip(axes, panels):
...     image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
...     _ = ax.set_xticks(range(len(cols)), cols)
...     _ = ax.set_yticks(range(len(rows)), rows)
...     _ = ax.set_title(title)
>>> _ = fig.colorbar(image, ax=axes, shrink=0.8)
>>> fig.savefig("mle.png", dpi=150)
```

![Two heatmaps of conditional probabilities for eight words: letters predicting sounds on the left, sounds predicting letters on the right.](../figures/mle.png)

Each row of the left panel sums to 1: the letter `c` splits between /k/ and
/s/, while most other letters map to a single sound. In the right panel, /k/
splits among `c` and `k`. The figure is produced by this code when the
documentation is built.

## Tips

- **Fix the colour scale** (`vmin`, `vmax`) when comparing directions or
  measures, so that the same colour means the same value.
- **Use a diverging colour map** (such as `"RdBu"`, centred on 0) for signed
  measures such as [PMI](../measures/pmi.md) and
  [the Tresoldi measure](../measures/tresoldi.md).
- **Mask unobserved pairs**, which get conventional scores (see
  [Pitfalls](../concepts/pitfalls.md#unobserved-pairs-are-scored-too)), for
  example with `numpy.ma.masked_where` on a matrix built from `scorer.obs`.
- **Order rows and columns** by frequency or by a clustering of the scores
  when the alphabets are large; the matrices follow the order of the alphabet
  lists, so reorder both together.
