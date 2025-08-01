#!/usr/bin/env python3
"""
Command-line interface for ASymCat.

Provides access to the main functionality of the ASymCat library for
analyzing categorical co-occurrence associations.
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tabulate  # type: ignore

import asymcat
from asymcat.scorer import CatScorer


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="ASymCat: Asymmetric measures of association for categorical variables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  asymcat data.tsv --scorers mle pmi chi2 --output results.json
  asymcat data.tsv --format pa-matrix --scorers all --table-format markdown
  asymcat data.tsv --ngrams 2 --scorers tresoldi theil_u --output-format csv
        """,
    )

    # Input options
    parser.add_argument("input", help="Input file path (TSV format)")
    parser.add_argument(
        "--format",
        choices=["sequences", "pa-matrix"],
        default="sequences",
        help="Input data format (default: sequences)",
    )
    parser.add_argument("--ngrams", type=int, help="N-gram order for sequence analysis")
    parser.add_argument("--pad", default="#", help="Padding symbol for n-grams (default: '#')")

    # Scoring options
    scorer_choices = [
        "mle",
        "pmi",
        "pmi_smoothed",
        "npmi",
        "chi2",
        "chi2_ns",
        "cramers_v",
        "cramers_v_ns",
        "fisher",
        "theil_u",
        "cond_entropy",
        "tresoldi",
        "mutual_information",
        "normalized_mutual_information",
        "jaccard_index",
        "goodman_kruskal_lambda",
        "log_likelihood_ratio",
        "all",
    ]
    parser.add_argument(
        "--scorers",
        nargs="+",
        choices=scorer_choices,
        default=["mle"],
        help="Scoring methods to compute (default: mle)",
    )

    # Smoothing options (for freqprob integration)
    parser.add_argument(
        "--smoothing",
        choices=["mle", "laplace", "lidstone"],
        default="mle",
        help="Smoothing method for probability estimation (default: mle)",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=1.0,
        help="Smoothing parameter (alpha for Laplace/Lidstone, default: 1.0)",
    )

    # Scaling options
    parser.add_argument(
        "--scale",
        choices=["minmax", "mean", "stdev"],
        help="Scale the scores using the specified method",
    )
    parser.add_argument("--invert", action="store_true", help="Invert the scaled scores")

    # Output options
    parser.add_argument("--output", "-o", help="Output file path (default: stdout)")
    parser.add_argument(
        "--output-format",
        choices=["json", "csv", "table"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--table-format",
        default="grid",
        help="Table format for tabulate (default: grid)",
    )
    parser.add_argument("--precision", type=int, default=4, help="Decimal precision (default: 4)")

    # Filtering options
    parser.add_argument("--min-count", type=int, help="Minimum co-occurrence count threshold")
    parser.add_argument("--top", type=int, help="Show only top N results by score")
    parser.add_argument(
        "--sort-by",
        choices=["xy", "yx", "pair"],
        default="xy",
        help="Sort results by X→Y score, Y→X score, or pair name (default: xy)",
    )

    # Verbosity
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return parser


def load_data(file_path: str, format_type: str, verbose: bool = False) -> list[list[list[str]]] | list[tuple]:
    """Load data from the input file."""
    if verbose:
        print(f"Loading data from {file_path} (format: {format_type})", file=sys.stderr)

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if format_type == "sequences":
        data = asymcat.read_sequences(file_path)
    elif format_type == "pa-matrix":
        data = asymcat.read_pa_matrix(file_path)  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown format: {format_type}")

    if verbose:
        print(f"Loaded {len(data)} data points", file=sys.stderr)

    return data


def compute_cooccurrences(
    data: list[list[list[str]]] | list[tuple], ngrams: int | None, pad: str, verbose: bool = False
) -> list[tuple]:
    """Compute co-occurrences from the data."""
    if verbose:
        if ngrams:
            print(f"Computing {ngrams}-gram co-occurrences", file=sys.stderr)
        else:
            print("Computing full-sequence co-occurrences", file=sys.stderr)

    cooccs = asymcat.collect_cooccs(data, order=ngrams, pad=pad)  # type: ignore[arg-type]

    if verbose:
        print(f"Found {len(cooccs)} co-occurrences", file=sys.stderr)

    return cooccs


def get_scorer_methods(scorer: CatScorer, scorer_names: list[str]) -> dict[str, Callable[..., Any]]:
    """Get the requested scoring methods from the scorer."""
    all_methods = {
        "mle": scorer.mle,
        "pmi": lambda: scorer.pmi(False),
        "pmi_smoothed": lambda: scorer.pmi_smoothed(False),
        "npmi": lambda: scorer.pmi(True),
        "chi2": lambda: scorer.chi2(True),
        "chi2_ns": lambda: scorer.chi2(False),
        "cramers_v": lambda: scorer.cramers_v(True),
        "cramers_v_ns": lambda: scorer.cramers_v(False),
        "fisher": scorer.fisher,
        "theil_u": scorer.theil_u,
        "cond_entropy": scorer.cond_entropy,
        "tresoldi": scorer.tresoldi,
        "mutual_information": scorer.mutual_information,
        "normalized_mutual_information": scorer.normalized_mutual_information,
        "jaccard_index": scorer.jaccard_index,
        "goodman_kruskal_lambda": scorer.goodman_kruskal_lambda,
        "log_likelihood_ratio": scorer.log_likelihood_ratio,
    }

    if "all" in scorer_names:
        return all_methods  # type: ignore[return-value]

    methods = {}
    for name in scorer_names:
        if name in all_methods:
            methods[name] = all_methods[name]
        else:
            raise ValueError(f"Unknown scorer: {name}")

    return methods  # type: ignore[return-value]


def compute_scores(
    cooccs: list[tuple],
    scorer_names: list[str],
    scale: str | None = None,
    invert: bool = False,
    min_count: int | None = None,
    smoothing_method: str = "mle",
    smoothing_alpha: float = 1.0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Compute the requested scores."""
    if verbose:
        print("Creating scorer and computing scores", file=sys.stderr)

    scorer = CatScorer(cooccs, smoothing_method=smoothing_method, smoothing_alpha=smoothing_alpha)
    methods = get_scorer_methods(scorer, scorer_names)

    results = {}
    for name, method in methods.items():
        if verbose:
            print(f"Computing {name}", file=sys.stderr)

        scores = method()

        # Apply scaling if requested
        if scale:
            if verbose:
                print(f"Scaling {name} using {scale}", file=sys.stderr)
            scores = asymcat.scorer.scale_scorer(scores, method=scale)

        # Apply inversion if requested
        if invert:
            if verbose:
                print(f"Inverting {name}", file=sys.stderr)
            scores = asymcat.scorer.invert_scorer(scores)

        # Apply minimum count filter if requested
        if min_count:
            filtered_scores = {}
            for pair, values in scores.items():
                if scorer.obs[pair]["11"] >= min_count:
                    filtered_scores[pair] = values
            scores = filtered_scores

        results[name] = scores

    return results


def format_output(
    results: dict[str, Any],
    output_format: str,
    table_format: str,
    precision: int,
    top: int | None = None,
    sort_by: str = "xy",
) -> str:
    """Format the results for output."""
    if output_format == "json":
        # Convert tuples to strings for JSON serialization
        json_results = {}
        for scorer_name, scores in results.items():
            json_results[scorer_name] = {str(pair): values for pair, values in scores.items()}
        return json.dumps(json_results, indent=2)

    elif output_format == "csv":
        lines = ["pair,scorer,score_xy,score_yx"]
        for scorer_name, scores in results.items():
            for pair, (score_xy, score_yx) in scores.items():
                lines.append(f'"{pair}",{scorer_name},{score_xy:.{precision}f},{score_yx:.{precision}f}')
        return "\n".join(lines)

    elif output_format == "table":
        if not results:
            return "No results to display."

        # Get all unique pairs
        all_pairs = set()
        for scores in results.values():
            all_pairs.update(scores.keys())

        # Sort pairs based on sort_by parameter
        if sort_by == "pair":
            sorted_pairs = sorted(all_pairs)
        else:
            first_scorer = list(results.keys())[0]
            if sort_by == "xy":
                sorted_pairs = sorted(all_pairs, key=lambda p: results[first_scorer].get(p, (0, 0))[0], reverse=True)
            elif sort_by == "yx":
                sorted_pairs = sorted(all_pairs, key=lambda p: results[first_scorer].get(p, (0, 0))[1], reverse=True)
            else:
                sorted_pairs = sorted(all_pairs)

        # Apply top N filter if requested
        if top:
            sorted_pairs = sorted_pairs[:top]

        # Build table headers
        headers = ["pair"]
        for scorer_name in results:
            headers.extend([f"{scorer_name}_xy", f"{scorer_name}_yx"])

        # Build table rows
        table_data = []
        for pair in sorted_pairs:
            row = [str(pair)]
            for scorer_name in results:
                if pair in results[scorer_name]:
                    score_xy, score_yx = results[scorer_name][pair]
                    row.extend([f"{score_xy:.{precision}f}", f"{score_yx:.{precision}f}"])
                else:
                    row.extend(["N/A", "N/A"])
            table_data.append(row)

        return tabulate.tabulate(table_data, headers=headers, tablefmt=table_format)

    else:
        raise ValueError(f"Unknown output format: {output_format}")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    if args.top is not None and args.top <= 0:
        print("Error: --top must be a positive integer", file=sys.stderr)
        sys.exit(1)

    if args.min_count is not None and args.min_count < 0:
        print("Error: --min-count must be non-negative", file=sys.stderr)
        sys.exit(1)

    if args.precision < 0:
        print("Error: --precision must be non-negative", file=sys.stderr)
        sys.exit(1)

    try:
        # Load data
        data = load_data(args.input, args.format, args.verbose)

        # Compute co-occurrences
        cooccs = compute_cooccurrences(data, args.ngrams, args.pad, args.verbose)

        if not cooccs:
            print("Error: No co-occurrences found in the data", file=sys.stderr)
            sys.exit(1)

        # Compute scores
        results = compute_scores(
            cooccs,
            args.scorers,
            args.scale,
            args.invert,
            args.min_count,
            args.smoothing,
            args.smoothing_alpha,
            args.verbose,
        )

        # Format output
        output = format_output(
            results,
            args.output_format,
            args.table_format,
            args.precision,
            args.top,
            args.sort_by,
        )

        # Write output
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            if args.verbose:
                print(f"Results written to {args.output}", file=sys.stderr)
        else:
            print(output)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
