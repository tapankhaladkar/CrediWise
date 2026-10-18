"""The project's charts, drawn the same way in the notebook and the README.

Needs matplotlib (a development dependency; the web service never imports
this module). Each function returns a figure and takes a theme, so the README
can serve a light and a dark version of every chart.

Colours: one series is blue; a second is orange. Both pairs (light and dark)
were checked for colour-blind separation and contrast against their surface.
Every chart also has a table twin (reports/metrics.json, docs/MODEL_CARD.md),
so no value depends on colour alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter
from sklearn.metrics import roc_auc_score, roc_curve

from . import config as cfg
from .explain import REASONS


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink_2: str
    muted: str
    grid: str
    axis: str
    reference: str
    blue: str
    orange: str

    def rc(self) -> dict:
        return {
            "figure.facecolor": self.surface, "axes.facecolor": self.surface,
            "savefig.facecolor": self.surface, "figure.dpi": 110,
            "font.family": "sans-serif", "font.size": 10.5,
            "text.color": self.ink, "axes.labelcolor": self.ink_2,
            "axes.titlecolor": self.ink, "axes.titlesize": 12,
            "axes.titleweight": "bold", "axes.titlelocation": "left",
            "axes.titlepad": 10, "axes.edgecolor": self.axis, "axes.linewidth": 0.8,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.color": self.grid, "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": self.muted, "ytick.color": self.muted,
            "xtick.labelcolor": self.ink_2, "ytick.labelcolor": self.ink_2,
            "legend.frameon": False, "legend.labelcolor": self.ink_2,
            "lines.linewidth": 2, "lines.solid_capstyle": "round",
        }


LIGHT = Theme("light", surface="#fcfcfb", ink="#0b0b0b", ink_2="#52514e",
              muted="#898781", grid="#e1e0d9", axis="#c3c2b7", reference="#c3c2b7",
              blue="#2a78d6", orange="#eb6834")
DARK = Theme("dark", surface="#1a1a19", ink="#f0efec", ink_2="#c3c2b7",
             muted="#898781", grid="#2c2c2a", axis="#383835", reference="#6b6a66",
             blue="#3987e5", orange="#d95926")

_PERCENT = PercentFormatter(1, decimals=0)
_COMPACT = FuncFormatter(lambda v, _: f"{v / 1000:g}K" if v >= 1000 else f"{v:g}")


def default_rate_by_lateness(df: pd.DataFrame, theme: Theme = LIGHT) -> Figure:
    """Past lateness against next-month default."""
    by_late = df.groupby("months_late")[cfg.TARGET].mean()
    overall = df[cfg.TARGET].mean()
    with plt.rc_context(theme.rc()):
        fig, ax = plt.subplots(figsize=(7.6, 3.9))
        ax.bar(by_late.index, by_late.values, width=0.3, color=theme.blue)
        ax.axhline(overall, color=theme.muted, linewidth=1)
        ax.text(6.55, overall, f"all customers\n{overall:.0%}", color=theme.ink_2,
                ha="left", va="center", fontsize=9.5, clip_on=False)
        for x in (by_late.index.min(), by_late.index.max()):
            ax.text(x, by_late[x] + 0.015, f"{by_late[x]:.0%}", ha="center",
                    color=theme.ink, fontsize=10)
        ax.set_title("The more months a customer paid late, the likelier they were to default")
        ax.set_xlabel("Months paid late, out of the last six")
        ax.set_ylabel("Defaulted the next month")
        ax.yaxis.set_major_formatter(_PERCENT)
        ax.set_ylim(0, by_late.max() * 1.15)
        ax.grid(axis="x", visible=False)
        fig.subplots_adjust(right=0.86)
    return fig


def roc(y, model_p, benchmark_p, theme: Theme = LIGHT) -> Figure:
    """Ranking quality of the model against the logistic-regression benchmark."""
    with plt.rc_context(theme.rc()):
        fig, ax = plt.subplots(figsize=(5.6, 5.4))
        ax.plot([0, 1], [0, 1], color=theme.reference, linewidth=1)
        ax.text(0.62, 0.55, "chance", color=theme.muted, fontsize=9.5, rotation=42)
        # Fixed colours per entity: the model is always blue, the benchmark orange.
        for p, color, name in [(benchmark_p, theme.orange, "Logistic regression"),
                               (model_p, theme.blue, "Gradient boosting")]:
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=color, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        ax.set_title("Gradient boosting flags more defaulters\nfor the same number of false alarms")
        ax.set_xlabel("Customers who repaid but were flagged")
        ax.set_ylabel("Defaulters flagged")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(_PERCENT)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right")
    return fig


def calibration(table: pd.DataFrame, theme: Theme = LIGHT) -> Figure:
    """Predicted against actual default rate, by probability bin."""
    with plt.rc_context(theme.rc()):
        fig, ax = plt.subplots(figsize=(5.6, 5.4))
        ax.plot([0, 1], [0, 1], color=theme.reference, linewidth=1, label="Perfect agreement")
        ax.plot(table["predicted"], table["observed"], color=theme.blue, marker="o",
                markersize=7, markeredgecolor=theme.surface, markeredgewidth=1.5,
                label="Model, test set")
        ax.set_title("Predicted risk matches what actually happened")
        ax.set_xlabel("Predicted chance of default")
        ax.set_ylabel("Share who actually defaulted")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(_PERCENT)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right")
    return fig


def decisions(y, decided, policy, theme: Theme = LIGHT) -> Figure:
    """Who lands in each decision band, split by what actually happened."""
    y, d = np.asarray(y), np.asarray(decided)
    bands = [("approve", f"Approve\n(risk under {policy.approve_below:.1%})"),
             ("refer", "Refer to\na person"),
             ("decline", f"Decline\n(risk {policy.decline_at:.0%}+)")]
    repaid = np.array([((d == b) & (y == 0)).sum() for b, _ in bands])
    defaulted = np.array([((d == b) & (y == 1)).sum() for b, _ in bands])
    approved_rate = defaulted[0] / (repaid[0] + defaulted[0])
    with plt.rc_context(theme.rc()):
        fig, ax = plt.subplots(figsize=(7.6, 3.4))
        pos = np.arange(len(bands))
        # The surface-coloured edge is the 2px gap between the two segments.
        ax.barh(pos, repaid, height=0.42, color=theme.blue, edgecolor=theme.surface,
                linewidth=1.0, label="Repaid")
        ax.barh(pos, defaulted, left=repaid, height=0.42, color=theme.orange,
                edgecolor=theme.surface, linewidth=1.0, label="Defaulted")
        total = repaid + defaulted
        for i in pos:
            ax.text(total[i] + total.max() * 0.015, i,
                    f"{defaulted[i] / total[i]:.0%} defaulted", va="center",
                    color=theme.ink, fontsize=10)
        ax.set_yticks(pos, [label for _, label in bands])
        ax.set_ylim(len(bands) - 0.4, -0.6)
        ax.set_xlim(0, total.max() * 1.28)
        ax.xaxis.set_major_formatter(_COMPACT)
        ax.set_xlabel("Customers in the test set")
        ax.set_title(f"Approved customers default at {approved_rate:.0%}, "
                     f"against {y.mean():.0%} if everyone were approved")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", ncols=2)
    return fig


def importance(table: pd.DataFrame, theme: Theme = LIGHT, top: int = 10) -> Figure:
    """Average size of each feature's SHAP contribution."""
    t = table.head(top)
    with plt.rc_context(theme.rc()):
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        ax.barh([REASONS[f] for f in t["feature"]][::-1], t["mean_abs_shap"][::-1],
                height=0.45, color=theme.blue)
        ax.set_title("Recent payment behaviour drives the predictions")
        ax.set_xlabel("Average influence on a prediction (mean |SHAP|, log-odds)")
        ax.grid(axis="y", visible=False)
    return fig


_FAIRNESS_PANELS = [
    ("sex", "Sex", ["female", "male"]),
    ("marital_status", "Marital status", ["married", "single"]),
    ("age_band", "Age", None),
    ("education", "Education", ["graduate_school", "university", "high_school"]),
]


def fairness(audit: dict, theme: Theme = LIGHT) -> Figure:
    """Approval rate among customers who repaid, per group."""
    panels = []
    for dim, label, order in _FAIRNESS_PANELS:
        g = pd.DataFrame(audit[dim]["by_group"]).set_index("group")
        g = g.drop(index=[i for i in g.index if i == "unknown"])
        g = g.reindex([o for o in order if o in g.index] if order else sorted(g.index))
        panels.append((label, g["approval_rate_if_repaid"]))
    # Each row is as tall as its largest panel needs, so every bar is the same
    # thickness without leaving half-empty panels.
    row_slots = [max(len(panels[0][1]), len(panels[1][1])),
                 max(len(panels[2][1]), len(panels[3][1]))]
    with plt.rc_context(theme.rc()):
        fig, axes = plt.subplots(2, 2, figsize=(9, 1.9 + 0.55 * sum(row_slots)), sharex=True,
                                 gridspec_kw={"height_ratios": row_slots})
        for i, (ax, (label, vals)) in enumerate(zip(axes.flat, panels)):
            ax.barh(range(len(vals)), vals.to_numpy(), height=0.45, color=theme.blue)
            ax.set_yticks(range(len(vals)), [n.replace("_", " ") for n in vals.index])
            ax.set_ylim(row_slots[i // 2] - 0.5, -0.6)
            for y, v in enumerate(vals):
                ax.text(v + 0.02, y, f"{v:.0%}", va="center", fontsize=9.5, color=theme.ink)
            ax.set_title(label, fontsize=11)
            ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(_PERCENT)
            ax.grid(axis="y", visible=False)
        fig.suptitle("Among customers who repaid, the share the model approved",
                     x=0.01, ha="left", fontsize=12, fontweight="bold", color=theme.ink)
        fig.tight_layout()
    return fig
