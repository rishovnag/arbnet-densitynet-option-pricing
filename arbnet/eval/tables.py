"""Result tables (LaTeX booktabs format)."""
from __future__ import annotations

from typing import Dict, List, Optional
import io


def format_results_table(
    results: Dict[str, Dict[str, float]],
    metric_keys: List[str],
    caption: str = "",
    label: str = "",
    metric_labels: Optional[Dict[str, str]] = None,
    model_order: Optional[List[str]] = None,
    n_digits: int = 4,
) -> str:
    """Render a model x metric results dict as a LaTeX booktabs table.

    Args:
        results: {model_name: {metric_key: value}}.
        metric_keys: which metrics to render (column order).
        caption, label: LaTeX caption/label.
        metric_labels: optional pretty names for metrics.
        model_order: optional row order.
        n_digits: precision.
    """
    if metric_labels is None:
        metric_labels = {k: k for k in metric_keys}
    if model_order is None:
        model_order = list(results.keys())
    buf = io.StringIO()
    cols = "l" + "r" * len(metric_keys)
    buf.write("\\begin{table}[t]\n\\centering\n")
    buf.write(f"\\caption{{{caption}}}\n")
    buf.write(f"\\label{{{label}}}\n")
    buf.write(f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n")
    header = "Model & " + " & ".join(metric_labels[k] for k in metric_keys) + " \\\\\n"
    buf.write(header)
    buf.write("\\midrule\n")
    for m in model_order:
        row_vals = []
        for k in metric_keys:
            v = results[m].get(k, float("nan"))
            if isinstance(v, float):
                row_vals.append(f"{v:.{n_digits}f}")
            else:
                row_vals.append(str(v))
        buf.write(f"{m} & " + " & ".join(row_vals) + " \\\\\n")
    buf.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return buf.getvalue()


def format_arbitrage_table(
    reports: Dict[str, dict],
    caption: str = "Static no-arbitrage violation rates by model",
    label: str = "tab:arb",
) -> str:
    """Render an arbitrage-violation table."""
    cols = ["butterfly_rate", "calendar_rate", "tail_rate", "total_rate"]
    labels = {
        "butterfly_rate": "Butterfly",
        "calendar_rate": "Calendar",
        "tail_rate": "Tail (Lee)",
        "total_rate": "Total",
    }
    return format_results_table(reports, cols, caption=caption, label=label, metric_labels=labels, n_digits=6)
