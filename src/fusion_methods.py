"""
Fusion Method
=================
Implementation of 15 classifier fusion methods across three levels:
- Abstract Level: Majority Vote, Weighted Majority Vote
- Rank Level: Borda Count, Highest Rank, Intersection, Union
- Measurement Level: Max, Min, Median, Sum, Product, Weighted Average,
                     Probabilistic Product, Decision Templates, Dempster-Shafer

"""

import numpy as np
from collections import Counter


# ============================================================
# ABSTRACT LEVEL FUSION
# ============================================================

def majority_vote(predictions):
    """Abstract level: Simple majority voting."""
    votes = list(predictions.values())
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]


def weighted_majority_vote(predictions, weights):
    """Abstract level: Weighted majority voting using classifier accuracies."""
    weighted_votes = Counter()
    for name, pred in predictions.items():
        weighted_votes[pred] += weights[name]
    return weighted_votes.most_common(1)[0][0]


# ============================================================
# RANK LEVEL FUSION
# ============================================================

def get_rank_matrix(probabilities):
    """Convert probability matrix to rank matrix (1 = most likely)."""
    ranks = {}
    for name, proba in probabilities.items():
        ranks[name] = np.argsort(-proba, axis=1)
    return ranks


def borda_count(probabilities, n_classes):
    """Rank level: Borda count method."""
    ranks = get_rank_matrix(probabilities)
    scores = np.zeros(n_classes)
    
    for name, rank_matrix in ranks.items():
        for class_idx in range(n_classes):
            rank_position = np.where(rank_matrix[0] == class_idx)[0][0]
            scores[class_idx] += (n_classes - rank_position)
    
    return np.argmax(scores)


def highest_rank(probabilities, n_classes):
    """Rank level: Select class with best (lowest) rank from any classifier."""
    ranks = get_rank_matrix(probabilities)
    best_class = None
    best_rank = n_classes
    
    for name, rank_matrix in ranks.items():
        top_class = rank_matrix[0, 0]
        if 0 < best_rank:  # rank 0 is best
            best_class = top_class
            best_rank = 0
    
    return best_class


def intersection_topk(probabilities, n_classes, k=10):
    """Rank level: Classes in top-k of ALL classifiers."""
    ranks = get_rank_matrix(probabilities)
    topk_sets = []
    
    for name, rank_matrix in ranks.items():
        topk_classes = set(rank_matrix[0, :k])
        topk_sets.append(topk_classes)
    
    intersection = set.intersection(*topk_sets)
    return intersection


def union_topk(probabilities, n_classes, k=1):
    """Rank level: Classes in top-k of ANY classifier."""
    ranks = get_rank_matrix(probabilities)
    union_set = set()
    
    for name, rank_matrix in ranks.items():
        topk_classes = set(rank_matrix[0, :k])
        union_set.update(topk_classes)
    
    return union_set


# ============================================================
# MEASUREMENT LEVEL FUSION
# ============================================================

def build_decision_profile(probabilities):
    """Build the Decision Profile matrix (L classifiers × C classes)."""
    dp = np.array(list(probabilities.values()))
    return dp


def max_rule(probabilities):
    """Measurement level: Max rule."""
    dp = build_decision_profile(probabilities)
    return np.argmax(np.max(dp, axis=0))


def min_rule(probabilities):
    """Measurement level: Min rule."""
    dp = build_decision_profile(probabilities)
    return np.argmax(np.min(dp, axis=0))


def median_rule(probabilities):
    """Measurement level: Median rule."""
    dp = build_decision_profile(probabilities)
    return np.argmax(np.median(dp, axis=0))


def sum_rule(probabilities):
    """Measurement level: Sum rule."""
    dp = build_decision_profile(probabilities)
    return np.argmax(np.sum(dp, axis=0))


def product_rule(probabilities, epsilon=0.001):
    """Measurement level: Product rule with epsilon smoothing."""
    dp = build_decision_profile(probabilities)
    dp[dp == 0] = epsilon
    return np.argmax(np.prod(dp, axis=0))


def weighted_average(probabilities, weights):
    """Measurement level: Weighted average using classifier accuracies."""
    dp = build_decision_profile(probabilities)
    weight_vector = np.array(list(weights.values()))
    weight_vector = weight_vector / weight_vector.sum()
    weighted_sum = np.average(dp, axis=0, weights=weight_vector)
    return np.argmax(weighted_sum)


def probabilistic_product(probabilities, class_priors, epsilon=0.001):
    """Measurement level: Probabilistic product with prior normalization."""
    dp = build_decision_profile(probabilities)
    dp[dp == 0] = epsilon
    n_classifiers = dp.shape[0]
    
    product = np.prod(dp, axis=0)
    prior_factor = class_priors ** (n_classifiers - 1)
    prior_factor[prior_factor == 0] = epsilon
    
    score = product / prior_factor
    return np.argmax(score)


def decision_templates(probabilities, X_train, y_train, models, n_classes):
    """Measurement level: Decision Templates method."""
    dp = build_decision_profile(probabilities)
    
    # Build templates from training data
    templates = np.zeros((n_classes, dp.shape[1]))
    for c in range(n_classes):
        mask = (y_train == c)
        if mask.sum() > 0:
            class_profiles = []
            for name, model in models.items():
                proba = model.predict_proba(X_train[mask])
                class_profiles.append(proba)
            templates[c] = np.mean(class_profiles, axis=0)
    
    # Compute similarity (1 - normalized Euclidean distance)
    similarities = np.zeros(n_classes)
    for c in range(n_classes):
        diff = dp - templates[c]
        similarities[c] = 1 - np.mean(diff ** 2)
    
    return np.argmax(similarities)


def dempster_shafer(probabilities, n_classes, epsilon=0.001):
    """Measurement level: Dempster-Shafer theory of evidence."""
    dp = build_decision_profile(probabilities)
    dp[dp == 0] = epsilon
    
    # Compute belief degrees
    beliefs = np.zeros((dp.shape[0], n_classes))
    for i in range(dp.shape[0]):
        total = dp[i].sum()
        beliefs[i] = dp[i] / total if total > 0 else dp[i]
    
    # Combine using Dempster's rule (product of beliefs)
    combined = np.prod(beliefs, axis=0)
    combined_sum = combined.sum()
    
    if combined_sum > 0:
        combined = combined / combined_sum
    
    return np.argmax(combined)
