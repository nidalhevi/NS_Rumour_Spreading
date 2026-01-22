# 11_statistical_analysis.py
# Statistical analysis for intervention experiments (IC & Threshold)
# - Robust baseline handling (auc_M_base / M_final_base)
# - Delta computation vs baseline for IC and TH
# - Within-configuration tests vs 0
# - Strategy contrasts vs Random
# - OLS trends over delay
# - Mixed-effects model (robust to dtype issues)
# - Heterogeneity by size/duration bins
# - Benefit-per-seed
# Outputs go to experiments/stats

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

print(">>> 11_statistical_analysis.py STARTED <<<")

# --------------------------------------------------------------------------------------------------
# Paths & IO
# --------------------------------------------------------------------------------------------------
ROOT = Path(".")
EXP_DIR = ROOT / "experiments"
TH_DIR  = ROOT / "experiments_threshold"
STATS_DIR = EXP_DIR / "stats"
STATS_DIR.mkdir(exist_ok=True, parents=True)

# Inputs from earlier steps
IC_PER_THREAD     = EXP_DIR / "intervention_results_per_thread.csv"
IC_SUMMARY        = EXP_DIR / "summary_by_strategy_tau_k.csv"
BASELINE_PER_THR  = EXP_DIR / "baseline_no_intervention.csv"       # from Step 7
TH_PER_THREAD     = TH_DIR / "intervention_threshold_per_thread.csv"  # from Step 9b
THREAD_DESC       = ROOT / "preprocessed" / "thread_descriptives.csv"  # for heterogeneity bins

# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------

def to_numeric(df, cols):
    """Coerce listed columns to numeric; ignore missing; return df."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def pick_col(df, candidates, label):
    """
    Return first column name from candidates that exists in df, else None with a warning.
    """
    for c in candidates:
        if c in df.columns:
            return c
    print(f"[WARN] Could not find a '{label}' column. Candidates tried: {candidates}. "
          f"Available: {list(df.columns)}")
    return None

def ensure_deltas(df, baseline_df, model_prefix="ic"):
    """
    Compute deltas vs baseline for per-thread data (IC or TH).
    Creates:
      - {prefix}_dAUC = (df auc)   - (baseline auc)
      - {prefix}_dM   = (df final) - (baseline final)
    Robust to various column namings:
      - baseline may have 'auc_M_base', 'M_final_base'
      - per-thread results may have 'auc_M_mean' or similar
    """
    if df is None or df.empty:
        return df

    if "thread_id" not in df.columns:
        print(f"[WARN] No thread_id in df ({model_prefix}); cannot merge baseline.")
        return df

    if baseline_df is None or baseline_df.empty:
        print(f"[WARN] No baseline provided for {model_prefix}; skipping delta computation.")
        return df

    # Short-circuit if deltas already present
    if all(c in df.columns for c in [f"{model_prefix}_dAUC", f"{model_prefix}_dM"]):
        return df

    # --- detect metric columns in df (results side) ---
    auc_candidates_df = [
        "auc_M_mean", "auc_mean", "auc", "AUC", "auc_M",
        "auc_M_obs", "auc_obs"
    ]
    fin_candidates_df = [
        "M_final_mean", "M_final", "final_mean", "final", "final_M"
    ]
    df_auc_col = pick_col(df, auc_candidates_df, f"{model_prefix} auc")
    df_fin_col = pick_col(df, fin_candidates_df, f"{model_prefix} final")
    if df_auc_col is None or df_fin_col is None:
        print(f"[WARN] Cannot compute deltas for {model_prefix}: missing metrics in df.")
        return df

    # --- detect baseline metric columns ---
    # Your baseline has: auc_M_base, M_final_base
    auc_candidates_base = ["auc_M_base", "auc_base", "auc", "auc_mean", "AUC"]
    fin_candidates_base = ["M_final_base", "final_base", "final", "final_mean", "final_M"]
    base_auc_col = pick_col(baseline_df, auc_candidates_base, "baseline auc")
    base_fin_col = pick_col(baseline_df, fin_candidates_base, "baseline final")
    if base_auc_col is None or base_fin_col is None:
        print(f"[WARN] Cannot compute deltas for {model_prefix}: missing metrics in baseline.")
        return df

    # Merge baseline
    base = baseline_df[["thread_id", base_auc_col, base_fin_col]].rename(
        columns={base_auc_col: "auc_base", base_fin_col: "final_base"}
    )
    out = df.merge(base, on="thread_id", how="left")

    # Numerics and deltas
    out = to_numeric(out, [df_auc_col, df_fin_col, "auc_base", "final_base"])
    out[f"{model_prefix}_dAUC"] = out[df_auc_col] - out["auc_base"]
    out[f"{model_prefix}_dM"]   = out[df_fin_col] - out["final_base"]
    return out

def ci95(arr):
    """95% CI half-width for a 1D array (ignoring NaNs)."""
    arr = pd.Series(arr).dropna().values
    if arr.size == 0:
        return np.nan
    # Normal approx is fine here (n is large). For robustness, one could use bootstrap.
    se = np.std(arr, ddof=1) / np.sqrt(arr.size)
    return 1.96 * se

# --------------------------------------------------------------------------------------------------
# Load Data
# --------------------------------------------------------------------------------------------------

ic_pt = pd.read_csv(IC_PER_THREAD) if IC_PER_THREAD.exists() else None
ic_sum = pd.read_csv(IC_SUMMARY) if IC_SUMMARY.exists() else None
baseline = pd.read_csv(BASELINE_PER_THR) if BASELINE_PER_THR.exists() else None
th_pt = pd.read_csv(TH_PER_THREAD) if TH_PER_THREAD.exists() else None
desc = pd.read_csv(THREAD_DESC) if THREAD_DESC.exists() else None

# Coerce some likely numeric cols up front
for df in [ic_pt, ic_sum, baseline, th_pt]:
    if df is not None:
        df.columns = [c.strip() for c in df.columns]  # trim whitespace
        numeric_guess = [c for c in df.columns if any(x in c.lower() for x in ["auc","final","peak","mean","std","runs","k","tau"])]
        to_numeric(df, numeric_guess)

# --------------------------------------------------------------------------------------------------
# Compute deltas vs baseline for IC and Threshold
# --------------------------------------------------------------------------------------------------

ic_pt = ensure_deltas(ic_pt, baseline, model_prefix="ic")
th_pt = ensure_deltas(th_pt, baseline, model_prefix="th")

# Save the per-thread with deltas (for reproducibility)
if ic_pt is not None and not ic_pt.empty:
    ic_pt.to_csv(STATS_DIR / "ic_per_thread_with_deltas.csv", index=False)
if th_pt is not None and not th_pt.empty:
    th_pt.to_csv(STATS_DIR / "th_per_thread_with_deltas.csv", index=False)

# --------------------------------------------------------------------------------------------------
# Within-configuration tests vs zero (IC deltas)
# --------------------------------------------------------------------------------------------------

print("[Step] Within-configuration tests vs 0 (IC Δ)")
ic_tests = []
if ic_pt is not None and not ic_pt.empty:
    # We expect columns: strategy, tau_min, k, ic_dAUC, ic_dM, thread_id, maybe veracity_bucket
    group_cols = ["strategy", "tau_min", "k"]
    for (s, tau, k), g in ic_pt.groupby(group_cols, observed=False):
        dauc = g["ic_dAUC"].dropna()
        dM = g["ic_dM"].dropna()
        if len(dauc) >= 2:
            t1, p1 = stats.ttest_1samp(dauc, 0.0, nan_policy="omit")
        else:
            t1, p1 = (np.nan, np.nan)
        if len(dM) >= 2:
            t2, p2 = stats.ttest_1samp(dM, 0.0, nan_policy="omit")
        else:
            t2, p2 = (np.nan, np.nan)
        ic_tests.append({
            "strategy": s, "tau_min": tau, "k": k,
            "N": len(g),
            "mean_dAUC": dauc.mean() if len(dauc) else np.nan,
            "ci95_dAUC": ci95(dauc),
            "t_dAUC": t1, "p_dAUC": p1,
            "mean_dM": dM.mean() if len(dM) else np.nan,
            "ci95_dM": ci95(dM),
            "t_dM": t2, "p_dM": p2
        })
ic_tests = pd.DataFrame(ic_tests)
ic_tests.to_csv(STATS_DIR / "ic_within_config_tests.csv", index=False)

# --------------------------------------------------------------------------------------------------
# Strategy vs Random contrasts (IC)
# --------------------------------------------------------------------------------------------------

print("[Step] Strategy vs Random contrasts (IC)")
contrasts = []
if ic_pt is not None and not ic_pt.empty:
    for (tau, k), g in ic_pt.groupby(["tau_min","k"], observed=False):
        g = g[["thread_id","strategy","ic_dAUC","ic_dM"]].dropna()
        # pivot per thread, compare to random
        pivot_auc = g.pivot_table(index="thread_id", columns="strategy", values="ic_dAUC", aggfunc="mean")
        pivot_m   = g.pivot_table(index="thread_id", columns="strategy", values="ic_dM", aggfunc="mean")
        if "random" not in pivot_auc.columns:
            continue
        for strat in pivot_auc.columns:
            if strat == "random":
                continue
            a = pivot_auc[strat]
            b = pivot_auc["random"]
            both = pd.concat([a, b], axis=1).dropna()
            if both.shape[0] >= 2:
                t_auc, p_auc = stats.ttest_rel(both.iloc[:,0], both.iloc[:,1], nan_policy="omit")
            else:
                t_auc, p_auc = (np.nan, np.nan)
            # final
            if strat in pivot_m.columns and "random" in pivot_m.columns:
                m1 = pivot_m[strat]
                m2 = pivot_m["random"]
                bothm = pd.concat([m1, m2], axis=1).dropna()
                if bothm.shape[0] >= 2:
                    t_m, p_m = stats.ttest_rel(bothm.iloc[:,0], bothm.iloc[:,1], nan_policy="omit")
                else:
                    t_m, p_m = (np.nan, np.nan)
            else:
                t_m, p_m = (np.nan, np.nan)
            contrasts.append({
                "tau_min": tau, "k": k, "strategy": strat,
                "N_pairs_auc": int(both.shape[0]),
                "t_auc_vs_random": t_auc, "p_auc_vs_random": p_auc,
                "N_pairs_final": int(bothm.shape[0]) if isinstance(bothm, pd.DataFrame) else 0,
                "t_final_vs_random": t_m, "p_final_vs_random": p_m
            })
contrasts = pd.DataFrame(contrasts)
contrasts.to_csv(STATS_DIR / "ic_strategy_vs_random_contrasts.csv", index=False)

# --------------------------------------------------------------------------------------------------
# OLS trends (IC): Δ vs delay (per strategy, budget)
# --------------------------------------------------------------------------------------------------

print("[Step] OLS trends (IC)")
ols_rows = []
if ic_pt is not None and not ic_pt.empty:
    ic_pt["tau_min"] = pd.to_numeric(ic_pt["tau_min"], errors="coerce")
    for (s, k), g in ic_pt.groupby(["strategy","k"], observed=False):
        tmp = g[["tau_min","ic_dAUC","ic_dM"]].dropna()
        if tmp.shape[0] >= 4:
            # Simple OLS: y ~ 1 + tau
            X = sm.add_constant(tmp["tau_min"].values.astype(float))
            y1 = tmp["ic_dAUC"].values.astype(float)
            y2 = tmp["ic_dM"].values.astype(float)
            try:
                m1 = sm.OLS(y1, X).fit()
                m2 = sm.OLS(y2, X).fit()
                ols_rows.append({
                    "strategy": s, "k": k,
                    "n": tmp.shape[0],
                    "slope_tau_on_dAUC": m1.params[1], "p_tau_on_dAUC": m1.pvalues[1],
                    "slope_tau_on_dM": m2.params[1],   "p_tau_on_dM": m2.pvalues[1],
                    "r2_dAUC": m1.rsquared, "r2_dM": m2.rsquared
                })
            except Exception as e:
                ols_rows.append({"strategy": s, "k": k, "error": str(e)})
ols_df = pd.DataFrame(ols_rows)
ols_df.to_csv(STATS_DIR / "ic_ols_trends.csv", index=False)

# --------------------------------------------------------------------------------------------------
# OLS model (IC): Δ ~ tau_min + k + (1|thread_id)
# --------------------------------------------------------------------------------------------------

print("[Step] OLS model (IC)")
mixed_rows = []
if ic_pt is not None and not ic_pt.empty:
    # Prepare a numeric-clean subset
    mdf = ic_pt.copy()
    keep = ["thread_id", "strategy", "tau_min", "k", "ic_dAUC", "ic_dM"]
    mdf = mdf[keep].dropna()
    mdf["tau_min"] = pd.to_numeric(mdf["tau_min"], errors="coerce")
    mdf["k"]       = pd.to_numeric(mdf["k"], errors="coerce")
    mdf["ic_dAUC"] = pd.to_numeric(mdf["ic_dAUC"], errors="coerce")
    mdf["ic_dM"]   = pd.to_numeric(mdf["ic_dM"], errors="coerce")
    mdf = mdf.dropna()
    mdf["tau_min_c"] = (mdf["tau_min"] - mdf["tau_min"].mean())/mdf["tau_min"].std()
    mdf["k_c"]       = (mdf["k"] - mdf["k"].mean())/mdf["k"].std()

    mdf["strategy"] = pd.Categorical(
        mdf["strategy"],
        categories=["random", "bridges", "hubs", "community", "earliest"],
        ordered=False
    )
    # Encode strategy as fixed effects (dummies)
    mdf = pd.get_dummies(mdf, columns=["strategy"], drop_first=True)

    try:
        # ΔAUC as outcome
        exog_cols = ["tau_min", "k"] + [c for c in mdf.columns if c.startswith("strategy_")]
        exog = sm.add_constant(mdf[exog_cols]).astype(float)
        endog = mdf["ic_dAUC"].astype(float)
        fe_auc = sm.OLS(endog, exog).fit(cov_type="cluster", cov_kwds={"groups": mdf["thread_id"]})
        with open(STATS_DIR / "fe_cluster_ic_dAUC.txt", "w") as f:
            f.write(fe_auc.summary().as_text())
    except Exception as e:
        print(f"[WARN] OLS ΔAUC failed: {e}")

    try:
        # ΔM as outcome
        exog_cols = ["tau_min", "k"] + [c for c in mdf.columns if c.startswith("strategy_")]
        exog = sm.add_constant(mdf[exog_cols]).astype(float)
        endog = mdf["ic_dM"].astype(float)
        fe_dm = sm.OLS(endog, exog).fit(cov_type="cluster", cov_kwds={"groups": mdf["thread_id"]})
        with open(STATS_DIR / "fe_cluster_ic_dM.txt", "w") as f:
            f.write(fe_dm.summary().as_text())
    except Exception as e:
        print(f"[WARN] OLS ΔM failed: {e}")

# --------------------------------------------------------------------------------------------------
# Benefit-per-seed (IC)
# --------------------------------------------------------------------------------------------------

print("[Step] Benefit-per-seed (IC)")
bps = []
if ic_pt is not None and not ic_pt.empty:
    ic_pt["bps_aucM"] = -ic_pt["ic_dAUC"] / pd.to_numeric(ic_pt["k"], errors="coerce")
    ic_pt["bps_finalM"] = -ic_pt["ic_dM"] / pd.to_numeric(ic_pt["k"], errors="coerce")
    bps = (ic_pt.groupby(["strategy","tau_min","k"], observed=False)[["bps_aucM","bps_finalM"]]
                .agg(["mean","median","count"])
                .reset_index())
    # flatten columns
    bps.columns = ['_'.join([c for c in col if c]).rstrip('_') for col in bps.columns.values]
    bps.to_csv(STATS_DIR / "ic_benefit_per_seed.csv", index=False)

# --------------------------------------------------------------------------------------------------
# IC vs TH paired comparisons (where available)
# --------------------------------------------------------------------------------------------------

print("[Step] IC vs Threshold paired comparisons")
ic_th_rows = []
if ic_pt is not None and not ic_pt.empty and th_pt is not None and not th_pt.empty:
    # Align on (thread, strategy, tau, k). First ensure consistent deltas exist in th_pt.
    # If th_dAUC/dM absent, try compute from raw columns (if any), else skip.
    th_has_d = ("th_dAUC" in th_pt.columns) and ("th_dM" in th_pt.columns)
    if not th_has_d:
        print("[WARN] TH deltas missing; attempting to infer deltas from TH columns (if present).")
        # Try to infer 'th_dAUC' from 'auc/th_auc' against baseline if ensure_deltas didn't do it
        th_pt = ensure_deltas(th_pt, baseline, model_prefix="th")

    th_has_d = ("th_dAUC" in th_pt.columns) and ("th_dM" in th_pt.columns)
    if not th_has_d:
        print("[WARN] No TH deltas found; IC vs TH comparison limited.")
    else:
        cols_keep = ["thread_id","strategy","tau_min","k","ic_dAUC","ic_dM"]
        i = ic_pt[cols_keep].dropna()
        t = th_pt[["thread_id","strategy","tau_min","k","th_dAUC","th_dM"]].dropna()
        j = i.merge(t, on=["thread_id","strategy","tau_min","k"], how="inner")
        if not j.empty:
            # simple paired differences IC - TH
            j["dAUC_diff"] = j["ic_dAUC"] - j["th_dAUC"]
            j["dM_diff"]   = j["ic_dM"]   - j["th_dM"]
            # overall
            for (s, tau, k), g in j.groupby(["strategy","tau_min","k"], observed=False):
                for col in ["dAUC_diff","dM_diff"]:
                    arr = g[col].dropna()
                    if len(arr) >= 2:
                        tstat, pval = stats.ttest_1samp(arr, 0.0)
                    else:
                        tstat, pval = (np.nan, np.nan)
                    ic_th_rows.append({
                        "strategy": s, "tau_min": tau, "k": k,
                        "metric": col, "mean": arr.mean() if len(arr) else np.nan,
                        "ci95": ci95(arr), "t": tstat, "p": pval, "N": len(arr)
                    })
        pd.DataFrame(ic_th_rows).to_csv(STATS_DIR / "ic_vs_th_paired.csv", index=False)

# --------------------------------------------------------------------------------------------------
# Heterogeneity by size/duration bins (IC)
# --------------------------------------------------------------------------------------------------

print("[Step] Heterogeneity (IC, by size/duration bins)")
if desc is not None and not desc.empty and ic_pt is not None and not ic_pt.empty:
    # Pull thread size and duration
    keep = ["thread_id","n_nodes","duration_min"]
    meta = desc[keep].drop_duplicates("thread_id")
    # Size bins and duration bins
    meta["size_bin"] = pd.qcut(meta["n_nodes"], q=3, labels=["small","medium","large"])
    meta["duration_bin"] = pd.qcut(meta["duration_min"], q=3, labels=["short","medium","long"])

    ic_delta = ic_pt.merge(meta, on="thread_id", how="left")
    # group by 4-way: strategy, tau, k, size_bin
    rows = []
    for (s, tau, k, sb), g in ic_delta.groupby(["strategy","tau_min","k","size_bin"], observed=False):
        dauc = g["ic_dAUC"].dropna()
        dM = g["ic_dM"].dropna()
        rows.append({
            "strategy": s, "tau_min": tau, "k": k, "size_bin": sb,
            "N": len(g),
            "mean_dAUC": dauc.mean() if len(dauc) else np.nan,
            "ci95_dAUC": ci95(dauc),
            "mean_dM": dM.mean() if len(dM) else np.nan,
            "ci95_dM": ci95(dM),
        })
    pd.DataFrame(rows).to_csv(STATS_DIR / "ic_heterogeneity_by_size.csv", index=False)

    rows = []
    for (s, tau, k, db), g in ic_delta.groupby(["strategy","tau_min","k","duration_bin"], observed=False):
        dauc = g["ic_dAUC"].dropna()
        dM = g["ic_dM"].dropna()
        rows.append({
            "strategy": s, "tau_min": tau, "k": k, "duration_bin": db,
            "N": len(g),
            "mean_dAUC": dauc.mean() if len(dauc) else np.nan,
            "ci95_dAUC": ci95(dauc),
            "mean_dM": dM.mean() if len(dM) else np.nan,
            "ci95_dM": ci95(dM),
        })
    pd.DataFrame(rows).to_csv(STATS_DIR / "ic_heterogeneity_by_duration.csv", index=False)

# --------------------------------------------------------------------------------------------------
# Write LaTeX stubs (examples)
# --------------------------------------------------------------------------------------------------

print("[Step] Writing LaTeX tables")

# Best-by-delta from ic_tests (like your Step 8 tables)
if not ic_tests.empty:
    # Rank by mean_dAUC (more negative is better)
    top_auc = ic_tests.sort_values("mean_dAUC").head(12)[
        ["strategy","tau_min","k","mean_dAUC","ci95_dAUC","mean_dM","ci95_dM","N"]
    ]
    top_auc.to_csv(STATS_DIR / "top12_by_dAUC.csv", index=False)
    with open(STATS_DIR / "top12_by_dAUC.tex", "w") as f:
        f.write(top_auc.to_latex(index=False, float_format="%.4g"))

    # Rank by mean_dM (more negative is better)
    top_final = ic_tests.sort_values("mean_dM").head(12)[
        ["strategy","tau_min","k","mean_dM","ci95_dM","mean_dAUC","ci95_dAUC","N"]
    ]
    top_final.to_csv(STATS_DIR / "top12_by_dM.csv", index=False)
    with open(STATS_DIR / "top12_by_dM.tex", "w") as f:
        f.write(top_final.to_latex(index=False, float_format="%.4g"))

print(">>> Statistical analysis COMPLETE <<<")
