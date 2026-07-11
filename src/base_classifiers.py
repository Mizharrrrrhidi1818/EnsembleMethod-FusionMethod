"""
Base Classifiers
===================
Training and evaluation of diverse base classifiers for ensemble fusion.

"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def initialize_base_classifiers():
    """Initialize a diverse set of base classifiers."""
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    return models


def train_all_models(models, X_train, y_train):
    """Train all base classifiers and record training accuracies."""
    trained_models = {}
    train_accuracies = {}
    
    print("\n" + "=" * 60)
    print("TRAINING BASE CLASSIFIERS")
    print("=" * 60)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        trained_models[name] = model
        train_accuracies[name] = train_acc
        print(f"{name:25s} | Training Accuracy: {train_acc:.4f}")
    
    print("=" * 60)
    return trained_models, train_accuracies


def get_predictions(models, X):
    """Get hard label predictions from all models."""
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X)
    return predictions


def get_probabilities(models, X):
    """Get probability predictions from all models."""
    probabilities = {}
    for name, model in models.items():
        probabilities[name] = model.predict_proba(X)
    return probabilities


def compute_shannon_entropy(probabilities):
    """Compute Shannon entropy for each sample across the ensemble."""
    avg_proba = np.mean(list(probabilities.values()), axis=0)
    entropy = -np.sum(avg_proba * np.log(avg_proba + 1e-12), axis=1)
    return entropy


def select_most_ambiguous_sample(models, X_test, y_test):
    """Select the test sample with maximum Shannon entropy."""
    probabilities = get_probabilities(models, X_test)
    entropy = compute_shannon_entropy(probabilities)
    ambiguous_idx = np.argmax(entropy)
    
    print(f"\nMost Ambiguous Sample:")
    print(f"  Index: {ambiguous_idx}")
    print(f"  True Class: {y_test[ambiguous_idx]}")
    print(f"  Entropy: {entropy[ambiguous_idx]:.4f}")
    
    return ambiguous_idx, X_test[ambiguous_idx:ambiguous_idx+1], y_test[ambiguous_idx]
