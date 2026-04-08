#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relative Weight Analysis (RWA) — 纯 numpy/pandas 实现
Johnson (2000) 方法：SVD 正交变换 → εj = Σ λjk² βk²

严格遵循 Johnson 原始论文公式：
  X = P Δ Q'  (SVD)
  Z = P Q'    (正交变换)
  Y = β₀ + Σβk Zk
  Xj = λ₀ + Σλjk Zk
  εj = Σ λjk² βk²

无外部依赖（仅需 numpy, pandas）
"""

import numpy as np
import pandas as pd
import sys
import os


def load_data(file_path):
    """Load CSV or Excel"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    raise ValueError("仅支持 CSV 和 Excel (.xlsx) 格式")


def prepare_data(df, dv, ivs):
    """
    Prepare data: select columns, drop NaN, separate X/y.
    Returns (X, y, ivs_used) or raises ValueError.
    """
    cols = [dv] + ivs
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据中不存在以下列: {missing}")

    sub = df[cols].dropna().copy()
    if len(sub) < 30:
        raise ValueError(f"有效样本量不足 (N={len(sub)})，建议至少 30+")

    # 数值化
    for c in sub.columns:
        sub[c] = pd.to_numeric(sub[c], errors='coerce')
    sub = sub.dropna()

    y = sub[dv].values.astype(float)
    X = sub[ivs].values.astype(float)
    return X, y, sub[ivs], len(sub)


def ols_r_squared(X, y):
    """OLS with intercept: y = a + X @ beta, returns R²"""
    n = X.shape[0]
    X_aug = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    y_hat = X_aug @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return max(0.0, 1 - ss_res / ss_tot)


def relative_weight_analysis(X, y):
    """
    Johnson (2000) RWA — 严格按论文公式实现.

    Steps:
    1. Z-score 标准化 X 和 y
    2. SVD: X_std = U @ diag(S) @ Vt  (即 X = P Δ Q')
    3. 正交变换: Z = U @ Vt  (即 Z = P Q')
    4. Y ~ Z 回归得到 βk
    5. 每个 Xj ~ Z 回归得到 λjk（因 Z 正交，λjk = corr(Xj, Zk)）
    6. εj = Σk λjk² βk²
    7. 相对权重 = εj / Σεj × 100%

    Input: X (n, p), y (n,)
    Returns: (results_df, overall_r2)
    """
    n, p = X.shape

    # Step 1: Z-score standardize (population std, ddof=0)
    X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
    y_std = (y - y.mean()) / y.std(ddof=0)

    # Step 2: SVD — X = P Δ Q'
    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)

    # Step 3: Orthogonal transformation — Z = P Q' = U @ Vt
    Z = U @ Vt  # (n, p)

    # Step 4: Y ~ Z regression → βk
    X_aug = np.column_stack([np.ones(n), Z])
    beta_full = np.linalg.lstsq(X_aug, y_std, rcond=None)[0]
    beta_k = beta_full[1:]  # (p,) — coefficients for each orthogonal component

    # Step 5: Xj ~ Z regression → λjk
    # Since Z columns are orthogonal, λjk = corr(Xj_std, Zk)
    lambda_jk = np.zeros((p, p))
    for j in range(p):
        for k in range(p):
            lambda_jk[j, k] = np.corrcoef(X_std[:, j], Z[:, k])[0, 1]

    # Step 6: εj = Σk λjk² βk²
    epsilon_j = np.zeros(p)
    for j in range(p):
        for k in range(p):
            epsilon_j[j] += lambda_jk[j, k] ** 2 * beta_k[k] ** 2

    # Overall R²
    overall_r2 = ols_r_squared(X_std, y_std)

    # Step 7: Rescaled relative weights (%)
    total = epsilon_j.sum()
    if total == 0:
        epsilon_j = np.ones(p) / p
        total = 1.0
    rel_weights = epsilon_j / total  # proportions
    rwa_r2 = epsilon_j / overall_r2  # standardized: εj / R² (matches relaimpo output)

    # Correlations of each IV with DV (for sign/direction)
    correlations = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(p)])

    # Build results
    results = pd.DataFrame({
        'variable': [f'var_{i}' for i in range(p)],
        'raw_weight': epsilon_j,
        'rwa_r2': rwa_r2,        # εj / R² — standardized relative weight
        'relative_weight': rel_weights,
        'pct': rel_weights * 100,
        'correlation_with_dv': correlations,
        'sign': np.sign(correlations),
    })

    results = results.sort_values('pct', ascending=False).reset_index(drop=True)

    return results, overall_r2


def one_line_summary(results, dv_name, overall_r2):
    """Generate a concise Chinese summary."""
    top = results.iloc[0]
    bottom = results.iloc[-1]
    items = []
    for _, r in results.iterrows():
        sign = '正向' if r['sign'] > 0 else '负向'
        items.append(f"{r['variable']}({r['pct']:.1f}%)")

    summary = (
        f"RWA 结果（R²={overall_r2:.3f}，N={len(results)}个驱动因素）："
        f"对 {dv_name} 影响最大的是 {top['variable']}"
        f"（{top['pct']:.1f}%），"
        f"最小的是 {bottom['variable']}"
        f"（{bottom['pct']:.1f}%），"
        f"排序：{' > '.join(items)}。"
    )
    return summary


def run_rwa(file_path, dv, ivs):
    """
    Full pipeline: load → prepare → analyze → summarize.
    Returns (results_df, overall_r2, summary, n).
    """
    df = load_data(file_path)
    X, y, iv_df, n = prepare_data(df, dv, ivs)

    results, overall_r2 = relative_weight_analysis(X, y)

    # Replace placeholder variable names with actual ones
    iv_list = list(iv_df.columns)
    var_map = {f'var_{i}': iv_list[i] for i in range(len(iv_list))}
    results['variable'] = results['variable'].map(var_map)

    summary = one_line_summary(results, dv, overall_r2)

    return results, overall_r2, summary, n


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Relative Weight Analysis (Johnson 2000)')
    parser.add_argument('--input', required=True)
    parser.add_argument('--dv', required=True, help='Dependent variable column name')
    parser.add_argument('--ivs', nargs='+', required=True, help='Independent variable column names')
    args = parser.parse_args()

    results, r2, summary, n = run_rwa(args.input, args.dv, args.ivs)
    print(f"样本量: {n}")
    print(f"总 R²: {r2:.4f}\n")
    print(results[['variable', 'raw_weight', 'rwa_r2', 'pct', 'correlation_with_dv']].to_string(index=False))
    print(f"\n{summary}")
