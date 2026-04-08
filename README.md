# Relative Weight Analysis (RWA) — OpenClaw Skill

纯 Python 实现的相对权重分析（Johnson 2000），用于确定多元回归中各自变量对因变量 R² 的相对贡献。

## 特性

- 🔬 **Johnson (2000) 原始方法**：SVD 正交变换 → εj = Σ λjk² βk²
- 📊 **高共线性场景**：专为因子间高度相关的用户研究场景设计
- 🚀 **零外部依赖**：仅 numpy + pandas
- 📋 **一键分析**：CLI + 库调用两种方式

## 快速使用

```bash
python3 scripts/rwa.py --input data.csv --dv satisfaction --ivs price quality service brand
```

## 方法说明

RWA 解决的核心问题：当自变量之间存在共线性时，传统回归系数无法准确反映各变量的相对重要性。

**算法步骤**：
1. Z-score 标准化
2. SVD：X = P Δ Q'
3. 正交变换：Z = P Q' = U @ Vt
4. Y ~ Z 回归 → βk
5. Xj ~ Z 回归 → λjk
6. εj = Σk λjk² βk²
7. 相对权重 = εj / Σεj × 100%

## 验证

已与 R 包 `relaimpo::relativeImp` 的结果对比验证，差异 <0.01%。

## 参考

- Johnson, J. W. (2000). A heuristic method for estimating the relative weight of predictor variables in multiple regression. *Multivariate Behavioral Research*, 35(1), 1-19.
- Tonidandel, S., & LeBreton, J. M. (2015). RWA relative weights package.

## 许可

MIT License
