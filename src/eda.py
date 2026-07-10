"""Generate the EDA report for the IEEE-CIS fraud dataset: class imbalance, missingness,
and the feature relationships most predictive of fraud. Writes figures to reports/figures/
and a markdown summary to reports/eda/eda_report.md.

Usage:
    .venv/Scripts/python.exe src/eda.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.load_data import load_merged  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
REPORT_PATH = ROOT / "reports" / "eda" / "eda_report.md"

# dataviz skill palette (references/palette.md) — categorical slot 1 (blue) / slot 6 (red)
COLOR_NONFRAUD = "#2a78d6"
COLOR_FRAUD = "#e34948"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": COLOR_GRID,
    "axes.labelcolor": COLOR_TEXT_SECONDARY,
    "axes.grid": True,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.8,
    "text.color": COLOR_TEXT,
    "xtick.color": COLOR_MUTED,
    "ytick.color": COLOR_MUTED,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def plot_class_imbalance(df: pd.DataFrame) -> dict:
    counts = df["isFraud"].value_counts().sort_index()
    total = counts.sum()
    non_fraud, fraud = counts.get(0, 0), counts.get(1, 0)
    fraud_rate = fraud / total

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Non-Fraud", "Fraud"], [non_fraud, fraud],
                   color=[COLOR_NONFRAUD, COLOR_FRAUD], width=0.6)
    for bar, count in zip(bars, [non_fraud, fraud]):
        ax.annotate(f"{count:,}\n({fmt_pct(count / total)})",
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, color=COLOR_TEXT)
    ax.set_ylabel("Transactions")
    ax.set_title("Class imbalance: fraud is a tiny minority", loc="left", fontsize=11, pad=20)
    ax.set_ylim(0, non_fraud * 1.18)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_imbalance.png", dpi=150)
    plt.close(fig)

    return {"total": int(total), "non_fraud": int(non_fraud), "fraud": int(fraud),
            "fraud_rate": fraud_rate}


def plot_missingness(df: pd.DataFrame, top_n: int = 20) -> pd.Series:
    missing = df.isna().mean().sort_values(ascending=False)
    top = missing.head(top_n)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.index[::-1], top.values[::-1], color=COLOR_MUTED)
    ax.set_xlabel("Fraction missing")
    ax.set_title(f"Top {top_n} features by missingness", loc="left", fontsize=11)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "missingness.png", dpi=150)
    plt.close(fig)
    return missing


def plot_amount_by_class(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [
        np.log1p(df.loc[df["isFraud"] == 0, "TransactionAmt"]),
        np.log1p(df.loc[df["isFraud"] == 1, "TransactionAmt"]),
    ]
    bp = ax.boxplot(data, tick_labels=["Non-Fraud", "Fraud"], patch_artist=True,
                     showfliers=False, widths=0.5)
    for patch, color in zip(bp["boxes"], [COLOR_NONFRAUD, COLOR_FRAUD]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor(color)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color(COLOR_TEXT_SECONDARY)
    ax.set_ylabel("log(1 + TransactionAmt)")
    ax.set_title("Transaction amount by class", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "amount_by_class.png", dpi=150)
    plt.close(fig)


def plot_fraud_rate_by_category(df: pd.DataFrame, col: str, min_count: int = 500,
                                 top_n: int = 10) -> pd.DataFrame:
    grp = df.groupby(col, dropna=False)["isFraud"].agg(["mean", "count"])
    grp = grp[grp["count"] >= min_count].sort_values("mean", ascending=False).head(top_n)
    overall_rate = df["isFraud"].mean()

    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(grp))))
    colors = [COLOR_FRAUD if v > overall_rate else COLOR_NONFRAUD for v in grp["mean"]]
    ax.barh([str(i) for i in grp.index][::-1], grp["mean"].values[::-1], color=colors[::-1])
    ax.axvline(overall_rate, color=COLOR_MUTED, linestyle="--", linewidth=1)
    ax.annotate(f"overall {fmt_pct(overall_rate)}", xy=(overall_rate, 0), xytext=(4, -14),
                textcoords="offset points", fontsize=8, color=COLOR_MUTED)
    ax.set_xlabel("Fraud rate")
    ax.set_title(f"Fraud rate by {col} (min {min_count} txns)", loc="left", fontsize=11)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fraud_rate_by_{col}.png", dpi=150)
    plt.close(fig)
    return grp


def plot_fraud_rate_over_time(df: pd.DataFrame) -> None:
    day = (df["TransactionDT"] // (24 * 3600)).astype(int)
    grp = df.groupby(day)["isFraud"].mean()

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(grp.index, grp.values, color=COLOR_FRAUD, linewidth=2)
    ax.set_xlabel("Day since first transaction")
    ax.set_ylabel("Fraud rate")
    ax.set_title("Fraud rate over time (potential concept drift signal)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fraud_rate_over_time.png", dpi=150)
    plt.close(fig)


def plot_top_correlations(df: pd.DataFrame, top_n: int = 15) -> pd.Series:
    numeric = df.select_dtypes(include=[np.number]).drop(columns=["isFraud", "TransactionID"],
                                                           errors="ignore")
    corr = numeric.corrwith(df["isFraud"]).dropna().sort_values(key=np.abs, ascending=False)
    top = corr.head(top_n)

    fig, ax = plt.subplots(figsize=(6, 0.4 * top_n + 1))
    colors = [COLOR_NONFRAUD if v >= 0 else COLOR_FRAUD for v in top.values]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.axvline(0, color=COLOR_MUTED, linewidth=1)
    ax.set_xlabel("Correlation with isFraud")
    ax.set_title(f"Top {top_n} numeric features correlated with fraud", loc="left", fontsize=11)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_correlations.png", dpi=150)
    plt.close(fig)
    return top


def write_report(stats: dict, missing: pd.Series, product_cd: pd.DataFrame,
                  card4: pd.DataFrame, device: pd.DataFrame, corr: pd.Series) -> None:
    lines = [
        "# EDA Report — IEEE-CIS Fraud Detection",
        "",
        "## Class imbalance",
        "",
        f"- Total transactions: {stats['total']:,}",
        f"- Non-fraud: {stats['non_fraud']:,} ({fmt_pct(1 - stats['fraud_rate'])})",
        f"- Fraud: {stats['fraud']:,} ({fmt_pct(stats['fraud_rate'])})",
        "",
        "Fraud is under 4% of all transactions — a naive model predicting \"not fraud\" "
        "for everything would score >96% accuracy while catching zero fraud. This is why "
        "the project optimizes for PR-AUC / precision-recall tradeoffs instead of accuracy.",
        "",
        "![Class imbalance](../figures/class_imbalance.png)",
        "",
        "## Missingness",
        "",
        f"- {(missing > 0.9).sum()} features are >90% missing.",
        f"- {(missing > 0.5).sum()} features are >50% missing.",
        "- Most missingness comes from the `id_*`/`D*`/`V*` feature blocks and identity fields "
        "that are only populated for a subset of transactions — this is informative (missingness "
        "itself can correlate with fraud) rather than pure noise, so it's preserved as a signal "
        "(e.g. via LightGBM's native NaN handling) rather than aggressively imputed.",
        "",
        "![Missingness](../figures/missingness.png)",
        "",
        "## Transaction amount",
        "",
        "![Amount by class](../figures/amount_by_class.png)",
        "",
        "## Fraud rate by category",
        "",
        "![Fraud rate by ProductCD](../figures/fraud_rate_by_ProductCD.png)",
        "",
        "![Fraud rate by card4](../figures/fraud_rate_by_card4.png)",
        "",
        "![Fraud rate by DeviceType](../figures/fraud_rate_by_DeviceType.png)",
        "",
        "## Fraud rate over time",
        "",
        "Fraud rate is not stationary across the observed window, which motivates the drift "
        "monitoring module later in the project — a model trained on early data may not "
        "reflect the fraud patterns in later data.",
        "",
        "![Fraud rate over time](../figures/fraud_rate_over_time.png)",
        "",
        "## Top correlated numeric features",
        "",
        "Correlation is a coarse signal (fraud is driven by nonlinear feature interactions "
        "that LightGBM/XGBoost captures far better than linear correlation), but it's a useful "
        "first pass for sanity-checking which engineered `C*`/`V*`/`D*` features carry signal.",
        "",
        "![Top correlations](../figures/top_correlations.png)",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading merged train data...")
    df = load_merged("train")

    print("Class imbalance...")
    stats = plot_class_imbalance(df)
    print(f"  fraud rate: {fmt_pct(stats['fraud_rate'])}")

    print("Missingness...")
    missing = plot_missingness(df)

    print("Transaction amount by class...")
    plot_amount_by_class(df)

    print("Fraud rate by category...")
    product_cd = plot_fraud_rate_by_category(df, "ProductCD")
    card4 = plot_fraud_rate_by_category(df, "card4")
    device = plot_fraud_rate_by_category(df, "DeviceType")

    print("Fraud rate over time...")
    plot_fraud_rate_over_time(df)

    print("Top correlations...")
    corr = plot_top_correlations(df)

    print("Writing report...")
    write_report(stats, missing, product_cd, card4, device, corr)
    print(f"Done. Report at {REPORT_PATH}, figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
