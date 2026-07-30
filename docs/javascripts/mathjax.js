// MathJax configuration for MkDocs-Material + pymdownx.arithmatex (generic mode).
// arithmatex emits \( ... \) and \[ ... \] wrapped in `.arithmatex`; MathJax
// renders those, and re-typesets on Material's instant-loading navigation.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
