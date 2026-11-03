# Prepare co-occurrence data

`CatScorer` takes a list of co-occurrences: pairs $(x, y)$ of values observed
together. This guide shows how to build that list from the usual shapes of
data.

## Two columns of a table

When each observation has a value for both variables, as two columns of a
table, pair them with `zip`:

```pycon
>>> from asymcat import CatScorer
>>> odour = ["almond", "anise", "none", "pungent", "none", "foul"]
>>> edible = ["yes", "yes", "yes", "no", "no", "no"]
>>> pairs = list(zip(odour, edible))
>>> pairs[:2]
[('almond', 'yes'), ('anise', 'yes')]
>>> CatScorer(pairs).mle()[("pungent", "no")]
(1.0, 0.3333333333333333)
```

With pandas, `list(zip(df["odour"], df["class"]))` does the same.

## Aligned sequences

For pairs of sequences whose elements correspond position by position, such as
a spelling and its pronunciation aligned symbol by symbol, use
`collect_cooccs` with `order=1`. Sequences can be strings or lists:

```pycon
>>> import asymcat
>>> aligned = [("cat", "kat"), (["c", "e", "n", "t"], ["s", "e", "n", "t"])]
>>> asymcat.collect_cooccs(aligned, order=1)
[('c', 'k'), ('a', 'a'), ('t', 't'), ('c', 's'), ('e', 'e'), ('n', 'n'), ('t', 't')]
```

The two sequences of a pair must have the same length. Alignment itself, such
as inserting gaps where a letter is silent, is up to you; a gap symbol such as
`"-"` is a value like any other.

## Contexts: n-grams

With a larger `order`, `collect_cooccs` pairs every element of an n-gram of one
sequence with every element of the corresponding n-gram of the other, so that
each symbol also co-occurs with its neighbours:

```pycon
>>> asymcat.collect_cooccs([("cat", "kat")], order=2)
[('c', 'k'), ('c', 'k'), ('c', 'a'), ('a', 'k'), ('a', 'a'), ('a', 'a'), ('a', 't'), ('t', 'a'), ('t', 't'), ('t', 't')]
```

The sequences are padded with `pad="#"` (by default) so that the first and last
elements occur in as many n-grams as the others, and pairs involving the pad
symbol are dropped. The pad symbol must not occur in the data.
`collect_ngrams` returns the n-grams themselves:

```pycon
>>> list(asymcat.collect_ngrams("cat", 2, "#"))
[('#', 'c'), ('c', 'a'), ('a', 't'), ('t', '#')]
```

## Whole sequences

Without `order`, `collect_cooccs` pairs every element of one sequence with
every element of the other, whatever their positions. This suits pairs of
unordered collections, such as the tags of a document and the words of its
title:

```pycon
>>> asymcat.collect_cooccs([(["red", "round"], ["apple"]), (["red"], ["cherry", "apple"])])
[('red', 'apple'), ('round', 'apple'), ('red', 'cherry'), ('red', 'apple')]
```

A pair of sequences of lengths $m$ and $k$ contributes $m \times k$
co-occurrences, so long sequences weigh more; see
[Pitfalls](../concepts/pitfalls.md#collecting-from-sequences-multiplies-the-evidence).

## Sequences from a file

`read_sequences` reads a tab-separated file with one pair of sequences per row
and elements separated by spaces, skipping the header row:

```pycon
>>> from pathlib import Path
>>> _ = Path("spelling.tsv").write_text(
...     "Orthography\tSegments\nc a t\tk a t\nc e n t\ts e n t\n",
...     encoding="utf-8",
... )
>>> data = asymcat.read_sequences("spelling.tsv")
>>> data
[[['c', 'a', 't'], ['k', 'a', 't']], [['c', 'e', 'n', 't'], ['s', 'e', 'n', 't']]]
>>> len(asymcat.collect_cooccs(data, order=1))
7
```

With `cols=["Orthography", "Segments"]`, it reads the named columns of a file
with more than two. `col_delim` and `elem_delim` change the separators.

## Presence–absence matrices

In a presence–absence matrix, each row is a site and each column records
whether a taxon (or feature) was observed there. `read_pa_matrix` turns it into
co-occurrences: one pair for every two taxa present at the same site. The file
needs an `ID` column for the sites:

```pycon
>>> _ = Path("islands.tsv").write_text(
...     "ID\tfinch\tgecko\tiguana\nnorth\t1\t1\t0\nsouth\t1\t1\t1\nwest\t0\t1\t1\n",
...     encoding="utf-8",
... )
>>> asymcat.read_pa_matrix("islands.tsv")
[('finch', 'gecko'), ('finch', 'gecko'), ('finch', 'iguana'), ('gecko', 'iguana'), ('gecko', 'iguana')]
```

Each pair is ordered alphabetically, so a taxon appears as $x$ or $y$ depending
on its name; the [Jaccard index](../measures/jaccard.md) and the other
symmetric measures are the natural choice for such data.
