---
name: relative-weight-analysis
description: >
  Perform Relative Weight Analysis (RWA / Johnson 2000) on user-uploaded CSV/Excel data to determine
  the relative importance of each independent variable in predicting a dependent variable. Outputs a
  ranked weight table and a one-line Chinese summary. Designed for user research and experience
  measurement (satisfaction drivers, NPS drivers, etc.).
  Triggers on: relative weight analysis, RWA, 相对权重分析, 因子重要性分析, 驱动因素分析,
  哪个因素最重要, 权重分析, 上传csv分析, 体验驱动因素.
---

# Relative Weight Analysis (RWA)

Analyze CSV/Excel data to determine each predictor's relative contribution to R².

## Workflow

When user uploads a CSV/Excel file and asks for RWA or driver analysis:

1. **Read the file** with `doc_parse` or `read`
2. **Identify columns**: Ask user which column is the dependent variable (DV) and which are independent variables (IVs). If unclear from context, present column names and ask.
3. **Run the script**:
   ```bash
   python3 scripts/rwa.py --input <file_path> --dv <dv_name> --ivs <iv1> <iv2> ...
   ```
4. **Deliver results**: Present the weight table + one-line summary directly in chat.

## Script

`scripts/rwa.py` — Pure numpy/pandas, no sklearn/statsmodels dependency.

**Input**:
- `--input`: Path to CSV or Excel file
- `--dv`: Dependent variable column name
- `--ivs`: Space-separated independent variable column names

**Output** (stdout):
- Sample size (N)
- Overall R²
- Weight table: variable, pct (%), correlation_with_dv, sign (+/-)
- One-line summary in Chinese

**Key requirements**:
- N ≥ 30 after dropping NaN
- All columns must be numeric
- Handles multicollinearity natively (PCA orthogonalization)

## One-Line Summary Format

```
RWA 结果（R²=0.476）：对 satisfaction 影响最大的因素是 quality（相对权重 43.5%，正向驱动），
各因素权重排序：quality(43.5%,正向) > brand(29.4%,正向) > price(18.6%,正向) > service(8.6%,正向)。
```

## Interpretation Quick Reference

| Weight % | Importance | Suggestion |
|----------|------------|------------|
| > 20% | Core driver | Top priority |
| 10-20% | Important driver | Key focus |
| 5-10% | Moderate | Optimize if resources allow |
| < 5% | Weak driver | Low priority |

## Edge Cases

- **N < 30**: Reject with message suggesting more data
- **Non-numeric columns**: Auto-convert via `pd.to_numeric`; if all NaN for a column, ask user to fix
- **Only 1 IV**: Still works but weight will be 100%
- **Highly collinear IVs**: RWA handles this gracefully (it's the method's strength)
- **R² < 0.1**: Mention that the model explains little variance; consider missing variables

## Method

Johnson, J. W. (2000). A heuristic method for estimating the relative weight of predictor variables in multiple regression. *Multivariate Behavioral Research*, 35(1), 1-19.

Algorithm: Standardize → SVD (X = U Δ Vt) → orthogonal transform (Z = U @ Vt) → regress Y on Z (βk) → regress Xj on Z (λjk) → εj = Σ λjk² βk² → normalize.

Validated against R package `relaimpo::relativeImp` with <0.01% difference.
