# robustness_compare_step10.py — Step 10: robustness sweeps & IC vs Threshold comparison
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EXP_IC_DIR  = Path("experiments")
EXP_TH_DIR  = Path("experiments_threshold")
OUT_DIR     = Path("experiments/robustness"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# Inputs from Step 7 & 9
IC_SUM = EXP_IC_DIR/"summary_by_strategy_tau_k.csv"
TH_SUM = EXP_TH_DIR/"intervention_threshold_aggregate.csv"

print(">>> robustness_compare_step10.py STARTED <<<")
ic = pd.read_csv(IC_SUM).rename(columns={
    "mean_dAUC_M":"ic_dAUC", "ci95_dAUC_M":"ic_ciAUC",
    "mean_dM_final":"ic_dM", "ci95_dM_final":"ic_ciM"})
th = pd.read_csv(TH_SUM).rename(columns={
    "auc_M_mean":"th_auc", "M_final_mean":"th_Mfinal"})

# For Threshold we need deltas vs a baseline (no-intervention). If you’ve run a baseline_th, merge it similarly.
# To keep this script self-contained, we’ll compare raw means; negative better in both.
cmp = (ic.merge(th, on=["strategy","tau_min","k"], how="inner"))

# --------- Plot: IC vs Threshold ΔAUC_M (or AUC)
plt.figure(figsize=(7,4), dpi=160)
for s in ["earliest","hubs","bridges","community","random"]:
    sub = cmp[cmp["strategy"]==s]
    # aggregate over k by averaging (or plot per k in small multiples if you prefer)
    for k in sorted(sub["k"].unique()):
        ss = sub[sub["k"]==k].sort_values("tau_min")
        x = ss["tau_min"].values
        y_ic = ss["ic_dAUC"].values
        y_th = -ss["th_auc"].values  # negate so "more negative is better" aligns visually
        plt.plot(x, y_ic, "-o", alpha=0.6, label=f"IC {s} k={k}")
        plt.plot(x, y_th, "--s", alpha=0.6, label=f"TH {s} k={k}")
plt.axhline(0, color="k", lw=0.8, alpha=0.5)
plt.xlabel("τ (minutes)"); plt.ylabel("Effect (negative better)")
plt.title("IC vs Threshold: effect vs delay (per strategy, per k)")
plt.legend(frameon=False, ncol=3, fontsize=8)
plt.tight_layout(); plt.savefig(OUT_DIR/"ic_vs_threshold_effect.png"); plt.close()

# --------- Benefit-per-seed (IC only; Threshold needs deltas vs baseline to be exact)
ic["bps_aucM"] = -ic["ic_dAUC"] / ic["k"]
plt.figure(figsize=(6.5,3.6), dpi=160)
for s in ["earliest","hubs","bridges","community","random"]:
    d = ic[ic["strategy"]==s]
    for k in sorted(d["k"].unique()):
        dd = d[d["k"]==k].sort_values("tau_min")
        plt.plot(dd["tau_min"], dd["bps_aucM"], marker="o", label=f"{s} k={k}")
plt.xlabel("τ (minutes)"); plt.ylabel("Benefit-per-seed (IC)")
plt.title("IC cost-effectiveness vs delay")
plt.legend(frameon=False, ncol=3, fontsize=8)
plt.tight_layout(); plt.savefig(OUT_DIR/"ic_bps_vs_tau.png"); plt.close()

# --------- Top cells table (IC) to LaTeX
top_ic = ic.sort_values("ic_dAUC").head(12)[["strategy","tau_min","k","ic_dAUC","ic_ciAUC","ic_dM","ic_ciM"]]
with open(OUT_DIR/"top_ic_cells.tex","w") as f:
    f.write("% Auto-generated\n")
    f.write(top_ic.rename(columns={"tau_min":"$\\tau$","k":"$k$"}).to_latex(index=False, float_format=lambda x: f"{x:.3g}"))

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
# Consistent colors per strategy
COL = {
    "earliest":  "#1f77b4",
    "hubs":      "#ff7f0e",
    "bridges":   "#2ca02c",
    "community": "#9467bd",
    "random":    "#7f7f7f",
}

# ---------- IC vs Threshold: effect vs delay (k=3 only) ----------
plt.figure(figsize=(7.5,4.2), dpi=160)

for s in ["earliest","hubs","bridges","community","random"]:
    sub = cmp[(cmp["strategy"]==s) & (cmp["k"]==3)].sort_values("tau_min")
    x = sub["tau_min"].values

    # IC styling: solid line + circle markers (hollow)
    y_ic = sub["ic_dAUC"].values
    plt.plot(
        x, y_ic, "-o", color=COL[s], lw=2,
        markerfacecolor="white", markeredgecolor=COL[s],
        markersize=6, markevery=1, label=f"IC — {s}"
    )

    # TH styling: dashed line + square markers (hollow)
    y_th = -sub["th_auc"].values
    plt.plot(
        x, y_th, "--s", color=COL[s], lw=2,
        markerfacecolor="white", markeredgecolor=COL[s],
        markersize=6, markevery=1, label=f"TH — {s}"
    )

plt.axhline(0, color="k", lw=0.8, alpha=0.5)
plt.xlabel("τ (minutes)")
plt.ylabel("Effect (negative better)")
plt.title("IC vs Threshold: effect vs delay (k=3)")
plt.grid(True, axis="y", alpha=0.2)
plt.legend(frameon=False, ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR/"ic_vs_threshold_effect_k3.png")
plt.close()

# ---------- Benefit-per-seed (IC only; k=3 only) ----------
ic["bps_aucM"] = -ic["ic_dAUC"] / ic["k"]
plt.figure(figsize=(7.0,3.8), dpi=160)

for s in ["earliest","hubs","bridges","community","random"]:
    d = ic[(ic["strategy"]==s) & (ic["k"]==3)].sort_values("tau_min")
    plt.plot(
        d["tau_min"], d["bps_aucM"],
        "-", marker="o", color=COL[s], lw=2,
        markerfacecolor="white", markeredgecolor=COL[s], markersize=6,
        label=s
    )

plt.xlabel("τ (minutes)")
plt.ylabel("Benefit-per-seed (−ΔAUC_M / k)")
plt.title("IC cost-effectiveness vs delay — k=3")
plt.grid(True, axis="y", alpha=0.2)
plt.legend(frameon=False, ncol=3, fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR/"ic_bps_vs_tau_k3.png")
plt.close()
#############################################################################################################################

print(f"Saved → {OUT_DIR}")
print(">>> Step 10 COMPLETE <<<")
