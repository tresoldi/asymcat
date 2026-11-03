# References

ASymCat implements established association measures from statistics,
information theory and computational linguistics. This page lists the primary
source for each one, so that work using ASymCat can cite the measures as well as
the software. All entries are also available as a BibTeX file:
[`references.bib`](references.bib).

To cite ASymCat itself, see [How to cite](citing.md).

## Probabilistic measures

`mle` reports conditional probabilities estimated from the co-occurrence counts.
With `smoothing_method="laplace"` or `"lidstone"`, the estimates come from
[FreqProb](https://freqprob.tresoldi.org/references/) (see
[Software](#software)).

- Lidstone, G. J. (1920). Note on the general case of the Bayes–Laplace formula
  for inductive or a posteriori probabilities. *Transactions of the Faculty of
  Actuaries*, 8, 182–192. (Laplace and Lidstone smoothing)

## Pointwise mutual information

`pmi` and `pmi_smoothed`; `pmi(normalized=True)` gives the normalized PMI.

- Church, K. W., & Hanks, P. (1990). Word association norms, mutual
  information, and lexicography. *Computational Linguistics*, 16(1), 22–29.
  [ACL Anthology J90-1003](https://aclanthology.org/J90-1003/)
- Bouma, G. (2009). Normalized (pointwise) mutual information in collocation
  extraction. In *Proceedings of the Biennial GSCL Conference 2009*
  (pp. 31–40).

## Statistical tests and contingency-table measures

`chi2`, `chi2_pvalue`, `cramers_v`, `fisher`, `fisher_pvalue`,
`log_likelihood_ratio`, `log_likelihood_ratio_pvalue` and
`goodman_kruskal_lambda`. On 2×2 tables, `chi2` uses SciPy's default Yates
continuity correction, and `cramers_v` is bias-corrected.

- Pearson, K. (1900). On the criterion that a given system of deviations from
  the probable in the case of a correlated system of variables is such that it
  can be reasonably supposed to have arisen from random sampling. *The London,
  Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, Series 5,
  50(302), 157–175.
  [doi:10.1080/14786440009463897](https://doi.org/10.1080/14786440009463897)
- Yates, F. (1934). Contingency tables involving small numbers and the χ² test.
  *Supplement to the Journal of the Royal Statistical Society*, 1(2), 217–235.
  [doi:10.2307/2983604](https://doi.org/10.2307/2983604)
- Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton, NJ:
  Princeton University Press. (Cramér's V)
- Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T.
  *Journal of the Korean Statistical Society*, 42(3), 323–328.
  [doi:10.1016/j.jkss.2012.10.002](https://doi.org/10.1016/j.jkss.2012.10.002)
  (the bias-corrected Cramér's V that `cramers_v` computes)
- Fisher, R. A. (1922). On the interpretation of χ² from contingency tables, and
  the calculation of P. *Journal of the Royal Statistical Society*, 85(1),
  87–94. [doi:10.2307/2340521](https://doi.org/10.2307/2340521)
- Dunning, T. (1993). Accurate methods for the statistics of surprise and
  coincidence. *Computational Linguistics*, 19(1), 61–74.
  [ACL Anthology J93-1003](https://aclanthology.org/J93-1003/) (the G²
  log-likelihood ratio)
- Goodman, L. A., & Kruskal, W. H. (1954). Measures of association for cross
  classifications. *Journal of the American Statistical Association*, 49(268),
  732–764.
  [doi:10.1080/01621459.1954.10501231](https://doi.org/10.1080/01621459.1954.10501231)
  (λ)

## Set-based measures

`jaccard_index`.

- Jaccard, P. (1912). The distribution of the flora in the alpine zone. *New
  Phytologist*, 11(2), 37–50.
  [doi:10.1111/j.1469-8137.1912.tb05611.x](https://doi.org/10.1111/j.1469-8137.1912.tb05611.x)

## Information-theoretic measures

`cond_entropy`, `mutual_information`, `normalized_mutual_information` and
`theil_u`. `normalized_mutual_information` divides mutual information by the
joint entropy, the complement of Rajski's distance.

- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System
  Technical Journal*, 27(3), 379–423.
  [doi:10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)
- Theil, H. (1970). On the estimation of relationships involving qualitative
  variables. *American Journal of Sociology*, 76(1), 103–154.
  [doi:10.1086/224909](https://doi.org/10.1086/224909) (the uncertainty
  coefficient)
- Rajski, C. (1961). A metric space of discrete probability distributions.
  *Information and Control*, 4(4), 371–377.
  [doi:10.1016/S0019-9958(61)80055-7](https://doi.org/10.1016/S0019-9958(61)80055-7)
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*
  (2nd ed.). Hoboken, NJ: Wiley.

### Bias-corrected entropy

`entropy_estimator="chao_shen"` and `entropy_estimator="nsb"`, computed by
FreqProb.

- Chao, A., & Shen, T.-J. (2003). Nonparametric estimation of Shannon's index of
  diversity when there are unseen species in sample. *Environmental and
  Ecological Statistics*, 10(4), 429–443.
  [doi:10.1023/A:1026096204727](https://doi.org/10.1023/A:1026096204727)
- Nemenman, I., Shafee, F., & Bialek, W. (2002). Entropy and inference,
  revisited. In *Advances in Neural Information Processing Systems 14*
  (pp. 471–478). MIT Press.
  [arXiv:physics/0108025](https://arxiv.org/abs/physics/0108025)

## The Tresoldi measure

`tresoldi` is specific to ASymCat; [The Tresoldi measure](measures/tresoldi.md)
gives its definition. It combines pointwise mutual information (Church & Hanks,
1990) with conditional probability.

## Resampling and multiple testing

`permutation_pvalue` shuffles the pairing of the two variables to estimate
significance; `bootstrap_ci` resamples the co-occurrences to give percentile
confidence intervals. [Test significance and uncertainty](how-to/significance.md#many-pairs-many-tests)
controls the false discovery rate when testing many pairs.

- Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *The
  Annals of Statistics*, 7(1), 1–26.
  [doi:10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552)
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A
  practical and powerful approach to multiple testing. *Journal of the Royal
  Statistical Society: Series B (Methodological)*, 57(1), 289–300.
  [doi:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)

## Software

ASymCat uses FreqProb for smoothing and entropy estimation, and is built on
NumPy and SciPy.

- Tresoldi, T. (2026). *FreqProb: A Python library for probability smoothing and
  frequency-based estimation* [Computer software]. Zenodo.
  [doi:10.5281/zenodo.23917393](https://doi.org/10.5281/zenodo.23917393)
  (all versions; ASymCat requires 0.8.0 or later, and FreqProb's
  [How to cite](https://freqprob.tresoldi.org/citing/) page gives the DOI of
  each version)
- Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array
  programming with NumPy. *Nature*, 585(7825), 357–362.
  [doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)
- Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0:
  Fundamental algorithms for scientific computing in Python. *Nature Methods*,
  17(3), 261–272.
  [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)

## Datasets

The sample datasets in the repository's
[`resources/`](https://github.com/tresoldi/asymcat/tree/master/resources)
directory come from third parties; their sources and licences are listed in
[`resources/README.md`](https://github.com/tresoldi/asymcat/blob/master/resources/README.md).
Cite the original source of any dataset you use.
