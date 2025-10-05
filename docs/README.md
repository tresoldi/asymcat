# ASymCat Documentation

Welcome to the ASymCat documentation! This directory contains comprehensive guides, tutorials, and references for the ASymCat library.

## 📚 Documentation Structure

### Core Documentation

| Document | Purpose | Best For |
|----------|---------|----------|
| **[User Guide](USER_GUIDE.md)** | Conceptual guide with theory and best practices | Understanding asymmetric association analysis, mathematical foundations, and methodology |
| **[API Reference](API_REFERENCE.md)** | Complete technical API documentation | Looking up specific functions, classes, and parameters |
| **[LLM Documentation](LLM_DOCUMENTATION.md)** | Practical guide for LLM coding agents | Quick integration, code examples, and common patterns |

### Interactive Tutorials (Nhandu)

Progressive, hands-on tutorials with executable code and visualizations:

1. **[Tutorial 1: Basics](tutorial_1_basics.py)** ([HTML](tutorial_1_basics.html))
   - Getting started with ASymCat
   - Basic workflow and data loading
   - Simple association measures
   - Understanding asymmetric relationships

2. **[Tutorial 2: Advanced Measures](tutorial_2_advanced_measures.py)** ([HTML](tutorial_2_advanced_measures.html))
   - Information-theoretic measures (PMI, MI, Theil U)
   - Statistical measures (Chi-square, Fisher)
   - Smoothing methods and their effects
   - Measure comparison and selection

3. **[Tutorial 3: Visualization](tutorial_3_visualization.py)** ([HTML](tutorial_3_visualization.html))
   - Heatmap visualizations
   - Score distributions and comparisons
   - Matrix transformations
   - Publication-quality plots

4. **[Tutorial 4: Real-World Applications](tutorial_4_real_world.py)** ([HTML](tutorial_4_real_world.html))
   - Linguistics: Grapheme-phoneme correspondence
   - Ecology: Species co-occurrence analysis
   - Market research: Product associations
   - Machine learning: Feature selection

> **Note:** Tutorials are written in Nhandu format (`.py` files with `#'` markdown comments) and can be executed to generate HTML reports with `make docs`.

## 🎯 Quick Navigation

### I want to...

**...understand what asymmetric association is**
→ Start with [User Guide - Introduction](USER_GUIDE.md#introduction)

**...get started quickly with code**
→ Check [LLM Documentation - Quick Start](LLM_DOCUMENTATION.md#quick-start) or [Tutorial 1](tutorial_1_basics.py)

**...look up a specific function**
→ Use [API Reference](API_REFERENCE.md)

**...understand the mathematical foundations**
→ Read [User Guide - Mathematical Background](USER_GUIDE.md#mathematical-background)

**...see visualizations and examples**
→ Run the [interactive tutorials](#interactive-tutorials-nhandu) or view their HTML outputs

**...integrate ASymCat into my project**
→ Follow [LLM Documentation - Common Patterns](LLM_DOCUMENTATION.md#common-patterns)

**...choose the right association measure**
→ See [User Guide - Association Measures Guide](USER_GUIDE.md#association-measures-guide) or [LLM Documentation - Measure Selection](LLM_DOCUMENTATION.md#measure-selection-decision-tree)

**...understand smoothing methods**
→ Read [User Guide - Smoothing Methods](USER_GUIDE.md#smoothing-methods) or [Tutorial 2](tutorial_2_advanced_measures.py)

**...see real-world applications**
→ Explore [Tutorial 4](tutorial_4_real_world.py) or [User Guide - Common Use Cases](USER_GUIDE.md#common-use-cases)

## 📖 Documentation Philosophy

### Three-Tier Approach

1. **User Guide** (Conceptual)
   - *"Why"* and *"when"* questions
   - Mathematical theory and intuition
   - Best practices and methodology
   - Domain-specific guidance

2. **API Reference** (Technical)
   - *"What"* and *"how"* questions
   - Complete function signatures
   - Parameter descriptions
   - Return types and examples

3. **LLM Documentation** (Practical)
   - *"Show me"* questions
   - Copy-paste code snippets
   - Common patterns and recipes
   - Quick integration examples

### Progressive Tutorials

Tutorials build skills incrementally:
- **Tutorial 1**: Foundation - basic workflow and concepts
- **Tutorial 2**: Depth - advanced measures and methods
- **Tutorial 3**: Communication - visualization techniques
- **Tutorial 4**: Application - real-world case studies

Each tutorial is self-contained but references previous concepts.

## 🔧 Working with Tutorials

### Viewing Tutorials

**HTML (Pre-generated):**
```bash
# Open in browser
firefox docs/tutorial_1_basics.html
```

**Python Source:**
```bash
# View source code
less docs/tutorial_1_basics.py
```

### Running Tutorials

**Execute and generate fresh HTML:**
```bash
# Regenerate all tutorial HTML files
make docs

# Or run individual tutorials
python docs/tutorial_1_basics.py
```

### Modifying Tutorials

1. Edit the `.py` file (Nhandu format with `#'` comments)
2. Run `make docs` to regenerate HTML
3. View updated HTML in browser

**Nhandu Syntax Reminder:**
```python
#' # This is a markdown header
#' Regular markdown text goes here.
#'
#' - Bullet point 1
#' - Bullet point 2

# This is a regular Python comment (not rendered)

import asymcat  # Code is executed and output captured

#' More markdown after code execution.
```

## 📊 Data Files

Tutorials use data files from `resources/`:
- `resources/linguistic_data.tsv` - Phoneme alignment sequences
- `resources/galapagos_finches.tsv` - Species presence-absence matrix
- `resources/market_data.tsv` - Customer purchase sequences
- Additional datasets for specific examples

## 🤝 Contributing to Documentation

When contributing:

1. **User Guide**: Add conceptual explanations, theory, best practices
2. **API Reference**: Keep synchronized with code (auto-generate if possible)
3. **LLM Documentation**: Add practical examples and common patterns
4. **Tutorials**: Ensure reproducibility, add visualizations, explain outputs

All documentation should:
- Be clear and concise
- Include working code examples
- Use consistent terminology
- Reference other documents for context

## 📄 Additional Resources

- **[Main README](../README.md)**: Project overview and quick start
- **[CHANGELOG](../CHANGELOG.md)**: Version history and migration guides
- **[CONTRIBUTING](../CONTRIBUTING.md)**: Contribution guidelines
- **[GitHub Repository](https://github.com/tresoldi/asymcat)**: Source code and issues

## 🔍 Search Tips

- **Function lookup**: Search API Reference for function names
- **Concept lookup**: Search User Guide for theoretical concepts
- **Example lookup**: Search LLM Documentation or tutorials
- **Error messages**: Check LLM Documentation troubleshooting section

---

**Need help?** Check the [User Guide](USER_GUIDE.md) for concepts, [API Reference](API_REFERENCE.md) for technical details, or [Tutorial 1](tutorial_1_basics.py) to get started!
