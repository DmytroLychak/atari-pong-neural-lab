"""Rebuild the figures directly from the submitted experiment records."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results"


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    rows = list(csv.DictReader((DATA / "training-episodes.csv").open()))
    x = np.array([int(r["timesteps"]) for r in rows])
    y = np.array([float(r["raw_return"]) for r in rows])
    smooth = np.array([y[max(0, i-49):i+1].mean() for i in range(len(y))])
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    ax.scatter(x / 1e6, y, s=6, color="0.7", label="Окремі навчальні партії")
    ax.plot(x / 1e6, smooth, color="black", lw=1.8, label="Середнє останніх 50 партій")
    ax.axvline(.700002, color="black", ls="--", lw=1, label="Обраний checkpoint")
    ax.set(xlabel="Зібрані переходи, млн", ylabel="Різниця очок у Pong", ylim=(-22, 22))
    ax.grid(alpha=.18)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(DATA / "training-curve.png", dpi=200)
    plt.close(fig)
    records = [json.loads((DATA / f).read_text()) for f in
               ["random-baseline-50.json", "initial-baseline-50.json", "final-evaluation.json"]]
    labels = ["Випадкові дії", "Ненавчена мережа", "Навчена PPO"]
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    bars = ax.bar(labels, [r["mean_return"] for r in records],
                  yerr=[r["std_return"] for r in records], capsize=5,
                  color="white", edgecolor="black", hatch=["//", "..", ""])
    for bar, r in zip(bars, records):
        ax.text(bar.get_x()+bar.get_width()/2, r["mean_return"] + (2.2 if r["mean_return"]>0 else -3.5),
                f"{r['mean_return']:+.2f}\n{r['wins']}/{r['episodes']} перемог", ha="center", va="center", fontsize=11)
    ax.axhline(0, color="black", lw=.7)
    ax.set(ylabel="Середня різниця очок ± SD", ylim=(-27, 24))
    ax.grid(axis="y", alpha=.18)
    fig.tight_layout()
    fig.savefig(DATA / "evaluation-chart.png", dpi=200)
    plt.close(fig)
    for filename in ["random-baseline-50", "initial-baseline-50", "final-evaluation"]:
        d=json.loads((DATA / (filename+".json")).read_text())
        with (DATA / (filename+".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer=csv.DictWriter(stream, fieldnames=list(d["games"][0]))
            writer.writeheader()
            writer.writerows(d["games"])
    print("Figures and per-game CSV files generated from experiment records.")


if __name__ == "__main__":
    main()
