#' # Tutorial 1: ASymCat Basics
#'
#' **Welcome to ASymCat!** This tutorial introduces the fundamentals of asymmetric categorical association analysis.
#'
#' ## What You'll Learn
#'
#' 1. What asymmetric associations are and why they matter
#' 2. Basic ASymCat workflow: load → collect → score
#' 3. Simple association measures (MLE, PMI, Jaccard)
#' 4. Understanding directional relationships
#' 5. Working with different data formats
#'
#' ## Prerequisites
#'
#' Basic Python knowledge and familiarity with probability concepts.

#' ## 1. Introduction: What Are Asymmetric Associations?
#'
#' Traditional association measures treat relationships as **symmetric**: the relationship
#' between X and Y is the same as between Y and X. But many real-world relationships
#' are **directional**:
#'
#' - **Medicine**: Knowing the doctor tells you the likely treatment, but knowing the
#'   treatment doesn't uniquely identify the doctor
#' - **Language**: The letter "q" strongly predicts "u" in English, but "u" weakly predicts "q"
#' - **Ecology**: One species' presence may predict another's, but not vice versa
#'
#' **ASymCat quantifies these directional relationships**, revealing patterns that
#' symmetric measures miss.

# Import required libraries
import asymcat
from asymcat.scorer import CatScorer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set up plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)

print("ASymCat version:", asymcat.__version__)
print("Ready to analyze asymmetric associations!\n")

#' ## 2. The Basic Workflow
#'
#' ASymCat analysis follows three simple steps:
#'
#' ```
#' 1. LOAD data    → read_sequences() or read_pa_matrix()
#' 2. COLLECT      → collect_cooccs() to count co-occurrences
#' 3. SCORE        → CatScorer() to compute association measures
#' ```
#'
#' Let's demonstrate with a simple example.

#' ### Example: Doctor Prescriptions
#'
#' We'll analyze a small dataset of doctor prescriptions to understand who prescribes what.

# Create a simple synthetic dataset
# Format: list of (X, Y) tuples representing co-occurrences
medical_data = [
    ('Dr_Smith', 'Aspirin'), ('Dr_Smith', 'Aspirin'), ('Dr_Smith', 'Aspirin'),
    ('Dr_Smith', 'Ibuprofen'),
    ('Dr_Jones', 'Morphine'), ('Dr_Jones', 'Morphine'), ('Dr_Jones', 'Morphine'),
    ('Dr_Brown', 'Aspirin'), ('Dr_Brown', 'Aspirin'),
    ('Dr_Brown', 'Ibuprofen'),
]

# Display the data
print("Medical Prescription Data:")
print("=" * 50)
df = pd.DataFrame(medical_data, columns=['Doctor', 'Treatment'])
print(df.to_string(index=False))
print(f"\nTotal prescriptions: {len(medical_data)}")
print(f"Unique doctors: {df['Doctor'].nunique()}")
print(f"Unique treatments: {df['Treatment'].nunique()}")

#' ### Step 1: Create a Scorer
#'
#' The `CatScorer` class is the heart of ASymCat. It takes co-occurrence data
#' and provides methods to compute various association measures.

# Create scorer (data is already in co-occurrence format)
scorer = CatScorer(medical_data, smoothing_method='laplace')

print("\nScorer created successfully!")
print(f"Smoothing method: {scorer.smoothing_method}")

#' ### Step 2: Compute Association Measures
#'
#' Let's start with **MLE (Maximum Likelihood Estimation)**, which computes
#' conditional probabilities: P(Y|X) and P(X|Y).

# Compute MLE scores
mle_scores = scorer.mle()

print("\nMLE (Conditional Probability) Results:")
print("=" * 60)

for (x, y), (p_y_given_x, p_x_given_y) in mle_scores.items():
    print(f"\nPair: {x} ↔ {y}")
    print(f"  P({y}|{x}) = {p_y_given_x:.3f}")
    print(f"  P({x}|{y}) = {p_x_given_y:.3f}")
    print(f"  Asymmetry = {abs(p_y_given_x - p_x_given_y):.3f}")

    if p_y_given_x > p_x_given_y:
        print(f"  → {x} better predicts {y}")
    else:
        print(f"  → {y} better predicts {x}")

#' ### Understanding the Results
#'
#' **Key Observations:**
#'
#' - **Dr_Smith → Aspirin**: P(Aspirin|Dr_Smith) = 0.75 means 75% of Dr_Smith's
#'   prescriptions are Aspirin
#' - **Aspirin → Dr_Smith**: P(Dr_Smith|Aspirin) = 0.60 means 60% of Aspirin
#'   prescriptions come from Dr_Smith
#' - The **asymmetry** shows these are different relationships!

#' ## 3. Visualizing Asymmetric Relationships
#'
#' Let's visualize the directional nature of these associations.

# Prepare data for visualization
pairs = list(mle_scores.keys())
pair_labels = [f"{x}\n↔\n{y}" for x, y in pairs]
xy_scores = [mle_scores[p][0] for p in pairs]
yx_scores = [mle_scores[p][1] for p in pairs]

# Create bar chart
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(pairs))
width = 0.35

bars1 = ax.bar(x_pos - width/2, xy_scores, width, label='X→Y', alpha=0.8, color='steelblue')
bars2 = ax.bar(x_pos + width/2, yx_scores, width, label='Y→X', alpha=0.8, color='coral')

ax.set_xlabel('Pairs', fontsize=12)
ax.set_ylabel('Conditional Probability P(·|·)', fontsize=12)
ax.set_title('MLE: Asymmetric Conditional Probabilities', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(pair_labels, fontsize=9)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.show()

print("\n📊 The different bar heights show the asymmetric nature of associations!")

#' ## 4. Multiple Association Measures
#'
#' ASymCat provides 17+ association measures. Let's compare a few basic ones:
#'
#' - **MLE**: Conditional probability P(Y|X)
#' - **PMI**: Pointwise Mutual Information (information overlap)
#' - **Jaccard**: Set overlap similarity

# Compute multiple measures
measures = {
    'MLE': scorer.mle(),
    'PMI': scorer.pmi(),
    'Jaccard': scorer.jaccard_index()
}

# Create comparison table
print("\nAssociation Measures Comparison:")
print("=" * 80)

comparison_data = []
for pair in pairs:
    x, y = pair
    row = {
        'X': x,
        'Y': y,
        'MLE(Y|X)': f"{measures['MLE'][pair][0]:.3f}",
        'MLE(X|Y)': f"{measures['MLE'][pair][1]:.3f}",
        'PMI(X,Y)': f"{measures['PMI'][pair][0]:.3f}",
        'Jaccard(X,Y)': f"{measures['Jaccard'][pair][0]:.3f}",
    }
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

#' ### Measure Properties
#'
#' | Measure | Range | Asymmetric? | Interpretation |
#' |---------|-------|-------------|----------------|
#' | **MLE** | [0,1] | ✓ Yes | Direct conditional probability |
#' | **PMI** | (-∞,+∞) | ✗ No | Information content (0 = independent) |
#' | **Jaccard** | [0,1] | ✗ No | Context overlap similarity |
#'
#' **Note:** Even symmetric measures like PMI return two values in ASymCat for
#' consistency, but both values are identical.

#' ## 5. Working with Sequence Data
#'
#' ASymCat can load aligned sequences from TSV files. Let's analyze English
#' orthography-to-pronunciation mappings to understand how letters map to sounds.

# Load sequence data from file
data = asymcat.read_sequences("resources/english_phonology.tsv")

print("\nEnglish Orthography → Phonology Data:")
print("=" * 60)
print(f"Number of words: {len(data)}")

# Display all word examples
for i, (letters, phones) in enumerate(data, 1):
    word = ''.join(letters)
    pronunciation = ' '.join(phones)
    print(f"\n{i}. {word:10s} → /{pronunciation}/")
    print(f"   Letters: {' '.join(letters)}")
    print(f"   Sounds:  {' '.join(phones)}")

#' ### Understanding the Examples
#'
#' These words demonstrate interesting orthography-phoneme patterns:
#'
#' - **FLEX**: Letter X maps to two sounds /k s/
#' - **GEES**: Letter G becomes /dʒ/, double E becomes /i/
#' - **SINGH**: Silent GH, NG cluster becomes /ŋ/
#' - **WIGHT**: Silent GH, I+GH digraph becomes /aɪ/
#' - **RAILS**: AI digraph becomes /eɪ/, S becomes /z/
#' - **SINGING**: Shows NG mapping to /ŋ/ twice, demonstrates morphological patterns
#' - **FLEET**: Double E becomes /i/, final T preserves /t/

#' ### Collecting Co-occurrences
#'
#' From aligned sequences, we extract co-occurrences at each position.
#' Each letter-sound pair becomes a data point for association analysis.
#'
#' **Note:** `collect_cooccs()` creates the Cartesian product of all elements
#' in each sequence pair. If you need aligned positions, sub-windows, or other
#' structures, preprocess your data before calling this function—ASymCat is
#' data-agnostic and treats each sequence pair independently.

# Collect co-occurrences
cooccs = asymcat.collect_cooccs(data)

print(f"\n\nTotal letter-sound co-occurrences: {len(cooccs)}")
print("\nAll co-occurrences:")
for i, (letter, sound) in enumerate(list(cooccs), 1):
    print(f"  {i:2d}. ({letter:1s}, {sound:3s})")

#' ### Analyzing Orthography-Phonology Associations
#'
#' Now we can quantify how predictable the letter-sound mappings are.

# Create scorer and analyze
seq_scorer = CatScorer(cooccs, smoothing_method='laplace')
seq_mle = seq_scorer.mle()

print("\n\nLetter → Sound Associations (Top 10 by P(Sound|Letter)):")
print("=" * 60)
sorted_pairs = sorted(seq_mle.items(), key=lambda x: x[1][0], reverse=True)
for i, ((letter, sound), (p_sound_letter, p_letter_sound)) in enumerate(sorted_pairs[:10], 1):
    print(f"{i:2d}. {letter} → {sound:3s}: P({sound}|{letter}) = {p_sound_letter:.3f}, "
          f"P({letter}|{sound}) = {p_letter_sound:.3f}")

#' ### Key Observations
#'
#' - **High P(Sound|Letter)**: When you see this letter, you reliably get this sound
#' - **Asymmetry**: P(Sound|Letter) ≠ P(Letter|Sound) reveals directional complexity
#' - **Example**: Letter 'S' might reliably map to /s/, but /s/ could come from
#'   'S', 'C', or 'SS' (lower reverse probability)

#' ## 6. Working with Presence-Absence Data
#'
#' ASymCat can also analyze binary presence-absence matrices, commonly used in ecology.

# Load Galápagos finch data
pa_data = asymcat.read_pa_matrix("resources/galapagos.tsv")

print("\nGalápagos Finch Dataset:")
print("=" * 50)
print(f"Co-occurrence pairs: {len(pa_data)}")

# Create scorer for species co-occurrence
species_scorer = CatScorer(pa_data, smoothing_method='laplace')
species_mle = species_scorer.mle()

print("\nTop 10 species associations (by P(Species2|Species1)):")
sorted_species = sorted(species_mle.items(), key=lambda x: x[1][0], reverse=True)

for i, ((sp1, sp2), (p_sp2_sp1, p_sp1_sp2)) in enumerate(sorted_species[:10], 1):
    # Shorten species names for display
    sp1_short = sp1.split()[-1] if '.' in sp1 else sp1
    sp2_short = sp2.split()[-1] if '.' in sp2 else sp2

    print(f"{i:2d}. {sp1_short:20s} → {sp2_short:20s}: P = {p_sp2_sp1:.3f}")

#' ## 7. Understanding Smoothing
#'
#' **Smoothing** handles sparse data by adding pseudo-counts. Let's see the effect.

# Create sparse dataset
sparse_data = [
    ('Common', 'Frequent'), ('Common', 'Frequent'), ('Common', 'Frequent'),
    ('Common', 'Rare'),
    ('Rare_Item', 'Frequent'),
]

print("\nSparse Dataset:")
for i, (x, y) in enumerate(sparse_data, 1):
    print(f"  {i}. ({x}, {y})")

# Compare different smoothing methods
smoothing_methods = {
    'MLE (no smoothing)': CatScorer(sparse_data, smoothing_method='mle'),
    'Laplace (α=1)': CatScorer(sparse_data, smoothing_method='laplace'),
    'Lidstone (α=0.1)': CatScorer(sparse_data, smoothing_method='lidstone', smoothing_alpha=0.1),
}

print("\nSmoothing Effects on P(Frequent|Common):")
print("=" * 50)

for method_name, smooth_scorer in smoothing_methods.items():
    scores = smooth_scorer.mle()
    if ('Common', 'Frequent') in scores:
        p_val = scores[('Common', 'Frequent')][0]
        print(f"{method_name:25s}: {p_val:.4f}")

#' **Interpretation:**
#'
#' - **MLE**: Raw probabilities (3/4 = 0.75) can be unstable for rare events
#' - **Laplace**: Adds 1 pseudo-count to all outcomes (smoother, more conservative)
#' - **Lidstone**: Allows custom α for fine-tuned smoothing
#'
#' **When to use smoothing:**
#' - Small datasets
#' - Rare co-occurrences
#' - When zero probabilities are problematic

#' ## 8. Key Takeaways
#'
#' **You've learned:**
#'
#' ✓ Asymmetric associations reveal directional relationships
#' ✓ Basic workflow: load → collect → score
#' ✓ MLE computes conditional probabilities P(Y|X) ≠ P(X|Y)
#' ✓ ASymCat handles sequences and presence-absence matrices
#' ✓ Smoothing stabilizes estimates for sparse data
#'
#' ## Next Steps
#'
#' Continue to **Tutorial 2: Advanced Measures** to learn:
#' - Information-theoretic measures (PMI, MI, Theil's U)
#' - Statistical measures (Chi-square, Fisher exact test)
#' - Choosing the right measure for your data
#' - Measure comparison and interpretation
#'
#' ## Quick Reference
#'
#' ```python
#' # Load sequence data
#' data = asymcat.read_sequences("file.tsv")
#' cooccs = asymcat.collect_cooccs(data)
#'
#' # Load presence-absence data
#' pa_data = asymcat.read_pa_matrix("matrix.tsv")
#'
#' # Create scorer
#' scorer = CatScorer(cooccs, smoothing_method='laplace')
#'
#' # Compute measures
#' mle = scorer.mle()           # Conditional probabilities
#' pmi = scorer.pmi()           # Pointwise mutual information
#' jaccard = scorer.jaccard_index()  # Set overlap
#' ```

print("\n" + "="*60)
print("Tutorial 1 Complete! ✓")
print("="*60)
print("\nYou're now ready to analyze asymmetric categorical associations!")
print("Proceed to Tutorial 2 to explore advanced measures.")
