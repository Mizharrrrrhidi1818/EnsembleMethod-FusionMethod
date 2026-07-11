# Ensemble Classifier Fusion Analysis

<div align="center">

**Multi-Level Ensemble Classifier Fusion on Amazon Sales Dataset**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Complete-green.svg)]()

</div>


## 🔍 Overview

This project implements a comprehensive **classifier fusion pipeline** that evaluates **15 different ensemble combination rules** across three distinct information levels on the Amazon Sales Dataset. The study investigates how different fusion strategies perform when base classifiers fundamentally disagree on ambiguous product samples.

### Fusion Levels Evaluated

| Level | Information Type | Methods |
|-------|-----------------|---------|
| **Abstract** | Hard class labels | Majority Vote, Weighted Majority Vote |
| **Rank** | Ordinal rankings | Borda Count, Highest Rank, Intersection, Union |
| **Measurement** | Probability vectors | Max, Min, Median, Sum, Product, Weighted Average, Probabilistic Product, Decision Templates, Dempster-Shafer |

---

## 💡 Motivation

Predictive modeling in e-commerce rarely relies on a single algorithm. Different classifiers capture distinct patterns:
- **Tree-based methods** handle non-linear feature interactions
- **Distance-based methods** excel in dense feature spaces
- **Linear models** provide stable decision boundaries

However, when these models **disagree** — especially on ambiguous or borderline samples — raw predictions from any individual model become unreliable.

**Classifier fusion** addresses this challenge by combining multiple model outputs into a single, more robust decision. This project answers:

> *When base classifiers are uncertain, which fusion strategy is most effective at recovering the correct classification?*

---

## 🎯 Objectives

1. ✅ Implement a complete fusion pipeline from raw data preprocessing through multi-level ensemble combination
2. ✅ Train diverse base classifiers (Decision Tree, Naive Bayes, KNN, Logistic Regression)
3. ✅ Identify the most ambiguous test sample using Shannon entropy
4. ✅ Extract predictions at three levels (abstract, rank, measurement)
5. ✅ Apply 15 fusion combination rules and evaluate their effectiveness
6. ✅ Analyze why fusion methods succeed or fail on extremely ambiguous cases


## 📊 Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | Kaggle — `karkavelrajaj/amazon-sales-dataset` |
| **Initial Shape** | 1,465 rows × 16 columns |
| **Final Shape** | ~1,463 rows × 9,139 features (after one-hot encoding) |
| **Target Variable** | `category` (product category) |
| **Number of Classes** | 56 (after rare-class filtering) |

### Preprocessing Steps
1. **Column Cleanup**: Dropped ID and URL columns
2. **Currency Parsing**: Converted ₹-formatted prices to floats
3. **Missing Value Imputation**: Median (numeric) and mode (categorical)
4. **Rare Class Filtering**: Removed classes with <6 samples
5. **Feature Encoding**: One-hot encoding (9,139 features)
6. **Feature Scaling**: StandardScaler (z-score normalization)
7. **Train/Test Split**: 70/30 stratified split

---


