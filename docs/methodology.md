# Methodology Documentation

## Ensemble Classifier Fusion Framework

### Overview
This project implements a comprehensive classifier fusion pipeline that evaluates 
15 different combination rules across three distinct information levels.

### Fusion Levels

#### 1. Abstract Level (Hard Labels)
- **Majority Vote**: Each classifier casts one vote; most votes wins
- **Weighted Majority Vote**: Votes weighted by classifier training accuracy

#### 2. Rank Level (Ordinal Positions)
- **Borda Count**: Classes receive points based on rank position
- **Highest Rank**: Class with best rank from any classifier wins
- **Intersection**: Classes in top-k of ALL classifiers
- **Union**: Classes in top-k of ANY classifier

#### 3. Measurement Level (Probability Scores)
- **Max Rule**: Highest single probability from any classifier
- **Min Rule**: Highest minimum probability (consensus)
- **Median Rule**: Median probability across classifiers
- **Sum Rule**: Sum of probabilities
- **Product Rule**: Product of probabilities (with epsilon smoothing)
- **Weighted Average**: Accuracy-weighted probability average
- **Probabilistic Product**: Product normalized by class priors
- **Decision Templates**: Euclidean similarity to class-specific templates
- **Dempster-Shafer**: Evidence-theoretic belief combination

### Ambiguity Selection
The most ambiguous test sample is selected using **Shannon entropy**:

$$H = -\sum_{i=1}^{C} p_i \log(p_i)$$

where $p_i$ is the average predicted probability for class $i$ across all classifiers.

### Base Classifiers
| Model | Algorithm | Purpose |
|-------|-----------|---------|
| Decision Tree | CART (max_depth=5) | Non-linear feature interactions |
| Naive Bayes | Gaussian | Probabilistic baseline |
| KNN | k=5 | Distance-based local patterns |
| Logistic Regression | L2 regularization | Linear decision boundaries |
