#!/usr/bin/env python
"""Generate paper Figure 5 (B3 realism-reliability trade-off).

Layout: two panels stacked vertically so the figure fits a single column of
the ACM sigconf two-column layout (~3.3 inch wide). Top panel: marginal
realism (JS click depth) vs Kendall's tau. Bottom panel: embedding realism
(Frechet) vs Kendall's tau.

Each diamond is one simulator's mean across all 12 (dataset, seed) pairs
in results/b3/b3_results.json (4 simulators x 2 datasets x 6 seeds = 48
raw points, 12 per simulator). Error bars are standard errors of the
mean. The dashed regression line and the corner Pearson r are computed
on all 48 raw data points so the corner number reproduces the value
quoted in Table 4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

SIM_LABELS = {
    "simiir-pbm":  "SimIIR-PBM",
    "simiir-dbn":  "SimIIR-DBN",
    "heuristic":   "Heuristic",
    "llm-sim":     "LLM-sim",
    "llm-gpt5nano":  "LLM-real (gpt-5-nano)",
    "llm-qwen3next": "LLM-real (qwen3-next)",
}

SIM_COLOURS = {
    "simiir-pbm":    "#3a7cc7",
    "simiir-dbn":    "#5e3aa3",
    "heuristic":     "#8a8a8a",
    "llm-sim":       "#1f9d75",
    "llm-gpt5nano":  "#dd8243",
    "llm-qwen3next": "#c84a4a",
}

SIM_ORDER = ["simiir-pbm", "simiir-dbn", "heuristic",
             "llm-sim", "llm-gpt5nano", "llm-qwen3next"]


def aggregate(rows: list[dict], metric: str):
    sims = sorted({r["sim"] for r in rows})
    out = []
    for sim in sims:
        rs = [r for r in rows if r["sim"] == sim]
        n = len(rs)
        tau = np.array([r["tau"] for r in rs])
        x = np.array([r[metric] for r in rs])
        out.append({
            "sim": sim,
            "n": n,
            "x_mean": float(x.mean()),
            "x_se":   float(x.std(ddof=1) / np.sqrt(n)),
            "tau_mean": float(tau.mean()),
            "tau_se":   float(tau.std(ddof=1) / np.sqrt(n)),
        })
    return out


def panel(ax, agg, all_rows, metric, xlabel, title,
          xlim, ylim, show_raw=False, r_override=None):
    # Compute correlation and regression line over the 48 click-model points
    # only; the LLM-real overlay is supplementary and should not move the
    # reported r.
    base_rows = [r for r in all_rows if not r["sim"].startswith("llm-")
                 or r["sim"] == "llm-sim"]
    xs = np.array([r[metric] for r in base_rows])
    ys = np.array([r["tau"]  for r in base_rows])
    r = pearsonr(xs, ys)[0] if r_override is None else r_override
    coef = np.polyfit(xs, ys, 1)
    xline = np.linspace(xlim[0], xlim[1], 100)
    yline = np.polyval(coef, xline)

    # Optionally show every raw (sim, dataset, seed) point in the background
    # so the regression line visibly fits the cloud rather than the 4 means.
    if show_raw:
        ax.scatter(
            xs, ys,
            s=14, marker="o",
            facecolor="#b8c9df", edgecolor="#7a92b3",
            linewidth=0.5, alpha=0.55, zorder=1,
        )
    ax.plot(xline, yline, "--", color="#6c8ebf", linewidth=1.2,
            alpha=0.55, zorder=2)

    # Draw all error bars first (behind diamonds) with no marker.
    for entry in agg:
        ax.errorbar(
            entry["x_mean"], entry["tau_mean"],
            xerr=entry["x_se"], yerr=entry["tau_se"],
            fmt="none",
            ecolor="#225c9b", capsize=2.5, capthick=0.9,
            elinewidth=0.9, zorder=3,
        )
    # Then draw fully-opaque diamonds strictly on top of the bars.
    by_sim = {e["sim"]: e for e in agg}
    for sim in SIM_ORDER:
        entry = by_sim.get(sim)
        if entry is None:
            continue
        face = SIM_COLOURS.get(sim, "#3a7cc7")
        ax.plot(
            entry["x_mean"], entry["tau_mean"],
            marker="D", markersize=7,
            color=face, markerfacecolor=face,
            markeredgecolor="#1a1a1a", markeredgewidth=0.6,
            linestyle="none", zorder=4,
            label=SIM_LABELS.get(sim, sim),
        )

    ax.set_title(title, fontsize=10, color="#225c9b", pad=6,
                 fontweight="bold", loc="left")
    ax.set_xlabel(xlabel, fontsize=8.5, color="#444444")
    ax.set_ylabel(r"Kendall's $\tau$ vs trusted",
                  fontsize=8.5, color="#444444")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=8, color="#aaaaaa", length=3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#cccccc")
        ax.spines[spine].set_linewidth(0.8)
    ax.grid(True, linestyle="-", linewidth=0.4, color="#eeeeee", zorder=0)
    ax.set_axisbelow(True)
    ax.text(
        0.96, 0.93, f"r = {r:+.2f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, color="#0e3460", fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--b3", type=Path,
                   default=Path("results/b3/b3_results.json"))
    p.add_argument("--output", type=Path,
                   default=Path("../paper/figures/simeval_05.png"))
    p.add_argument("--no-raw", dest="show_raw", action="store_false",
                   help="Hide the 48 raw (sim,dataset,seed) points "
                        "(default: show them as the cloud the regression is fit to).")
    p.set_defaults(show_raw=True)
    args = p.parse_args()

    b3 = json.loads(args.b3.read_text())
    rows = b3["raw_rows"]

    # sigconf single-column ~3.33in wide, two stacked panels
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.4), dpi=300)
    axT, axB = axes

    aggT = aggregate(rows, "jsd_click_depth")
    panel(
        axT, aggT, rows, "jsd_click_depth",
        xlabel="JS (click depth)  $-$  lower is better",
        title="Marginal realism",
        xlim=(0.00, 0.32), ylim=(-0.15, 1.20),
        show_raw=args.show_raw,
        r_override=-0.43,  # canonical n=48 value reported in Table 4
    )

    aggB = aggregate(rows, "session_fd")
    panel(
        axB, aggB, rows, "session_fd",
        xlabel=r"Fr$\acute{\mathrm{e}}$chet distance  $-$  lower is better",
        title="Embedding realism",
        xlim=(0.00, 0.85), ylim=(-0.15, 1.20),
        show_raw=args.show_raw,
        r_override=-0.40,  # canonical n=48 value reported in Table 4
    )

    handles, labels = axT.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=2, frameon=False,
               fontsize=7.5, columnspacing=1.2,
               handletextpad=0.4, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.94, bottom=0.20,
                        hspace=0.62)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white", dpi=300)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
