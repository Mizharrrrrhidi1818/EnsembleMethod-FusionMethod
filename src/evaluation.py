"""
Evaluation
=============
Evaluation and reporting module for ensemble classifier fusion results.

"""

import numpy as np
import pandas as pd


def evaluate_all_methods(results_dict, true_label):
    """Evaluate all fusion methods against the true label."""
    evaluation = []
    
    for method_name, decision in results_dict.items():
        if isinstance(decision, (set, list, np.ndarray)):
            correct = true_label in decision
        else:
            correct = (decision == true_label)
        
        evaluation.append({
            'Method': method_name,
            'Decision': str(decision),
            'Correct': correct
        })
    
    df_results = pd.DataFrame(evaluation)
    return df_results


def print_results_table(df_results):
    """Print formatted results table."""
    print("\n" + "=" * 70)
    print("FUSION METHODS EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Method':<35} {'Decision':<20} {'Correct':<10}")
    print("-" * 70)
    
    for _, row in df_results.iterrows():
        status = "✅" if row['Correct'] else "❌"
        print(f"{row['Method']:<35} {row['Decision']:<20} {status}")
    
    print("=" * 70)
    correct_count = df_results['Correct'].sum()
    total = len(df_results)
    print(f"Overall Accuracy: {correct_count}/{total} ({correct_count/total*100:.1f}%)")
    print("=" * 70)


def generate_summary_report(df_results, output_path="results/summary_report.md"):
    """Generate a Markdown summary report."""
    correct_count = df_results['Correct'].sum()
    total = len(df_results)
    
    report = f"""# Ensemble Classifier Fusion - Results Summary

## Overview
- **Total Methods Evaluated:** {total}
- **Correct Predictions:** {correct_count}
- **Overall Accuracy:** {correct_count/total*100:.1f}%

## Detailed Results

| # | Method | Decision | Correct? |
|---|--------|----------|----------|
"""
    
    for i, (_, row) in enumerate(df_results.iterrows(), 1):
        status = "✅" if row['Correct'] else "❌"
        report += f"| {i} | {row['Method']} | {row['Decision']} | {status} |\n"
    
    report += f"""
## Key Findings

1. All {total} fusion methods were evaluated on the most ambiguous test sample
2. The sample was selected using maximum Shannon entropy across the ensemble
3. Base classifiers showed significant disagreement on this sample
4. Measurement-level methods (Sum, Weighted Average, Dempster-Shafer) showed 
   the most nuanced probability aggregation

## Recommendations

- Ensure base model calibration before fusion
- Address curse of dimensionality with feature selection
- Consider reject options when fusion uncertainty is high
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Summary report saved to: {output_path}")
    return report
