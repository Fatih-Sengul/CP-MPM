#!/usr/bin/env python3
"""
Script to add remaining tier analysis cells to MAINTAINABILITY_PREDICTION.ipynb
This ensures all tiers are added in the correct order without manually inserting one by one.
"""

import json
import sys

# Load notebook
with open('MAINTAINABILITY_PREDICTION.ipynb', 'r') as f:
    notebook = json.load(f)

# Remove empty cells at the end
while notebook['cells'] and not notebook['cells'][-1].get('source'):
    notebook['cells'].pop()

# Define remaining tier cells
remaining_tiers = [
    {
        "name": "TIER 1.3",
        "source": """# =============================================================================
# TIER 1.3: CONFIDENCE CALIBRATION - Measuring Prediction Reliability
# =============================================================================

print("\\n" + "="*80)
print("TIER 1.3: CONFIDENCE CALIBRATION")
print("="*80 + "\\n")

print("📊 Evaluating how well prediction probabilities reflect true correctness likelihood\\n")

from sklearn.calibration import calibration_curve

# Use best model probabilities
if hasattr(best_model, 'predict_proba'):
    y_proba_calib = best_model.predict_proba(X_test_scaled)[:, 1]
else:
    print("⚠️  Model does not support probability predictions. Skipping calibration analysis.")
    y_proba_calib = None

if y_proba_calib is not None:
    print("1.3.1 Expected Calibration Error (ECE)")
    print("-" * 80 + "\\n")

    # Calculate ECE
    n_bins = 10
    prob_true, prob_pred = calibration_curve(y_test, y_proba_calib, n_bins=n_bins, strategy='uniform')

    # ECE = weighted average of |accuracy - confidence| across bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        # Get samples in this bin
        mask = (y_proba_calib >= bin_edges[i]) & (y_proba_calib < bin_edges[i+1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = mask | (y_proba_calib == 1.0)

        n_samples = mask.sum()
        if n_samples == 0:
            continue

        # Average confidence in bin
        conf = y_proba_calib[mask].mean()

        # Accuracy in bin
        acc = y_test[mask].mean()

        # Contribution to ECE
        ece += (n_samples / len(y_test)) * abs(acc - conf)

        bin_details.append({
            'Bin': f'[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})',
            'n_samples': int(n_samples),
            'Avg_Confidence': conf,
            'Accuracy': acc,
            'Calibration_Error': abs(acc - conf)
        })

    df_ece = pd.DataFrame(bin_details)

    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"  → {'Well-calibrated' if ece < 0.05 else 'Moderate' if ece < 0.15 else 'Poorly calibrated'}\\n")

    print("Calibration by Confidence Bin:\\n")
    print(df_ece.to_string(index=False, float_format='%.4f'))
    print()

    df_ece.to_csv(f'{OUTPUT_PATH}/tier1_calibration_bins.csv', index=False)
    print(f"✓ Calibration details saved: tier1_calibration_bins.csv\\n")

    # 1.3.2 Brier Score
    print("1.3.2 Brier Score Analysis")
    print("-" * 80 + "\\n")

    from sklearn.metrics import brier_score_loss

    brier = brier_score_loss(y_test, y_proba_calib)

    # Decompose Brier score (reliability, resolution, uncertainty)
    # Brier = Reliability - Resolution + Uncertainty
    # Using binning approach

    uncertainty = y_test.mean() * (1 - y_test.mean())

    # Resolution: how well model separates positive from negative
    bins = pd.cut(y_proba_calib, bins=10, duplicates='drop')
    bin_stats = pd.DataFrame({
        'prob': y_proba_calib,
        'actual': y_test,
        'bin': bins
    })

    resolution = 0
    reliability = 0

    for bin_val in bin_stats['bin'].unique():
        if pd.isna(bin_val):
            continue
        mask = bin_stats['bin'] == bin_val
        n_k = mask.sum()
        if n_k == 0:
            continue

        o_k = bin_stats.loc[mask, 'actual'].mean()  # Observed frequency in bin
        p_k = bin_stats.loc[mask, 'prob'].mean()    # Mean predicted probability in bin

        resolution += (n_k / len(y_test)) * (o_k - y_test.mean()) ** 2
        reliability += (n_k / len(y_test)) * (p_k - o_k) ** 2

    print(f"Brier Score: {brier:.4f}")
    print(f"  → Lower is better (0 = perfect, 0.25 = random for balanced data)\\n")

    print("Brier Score Decomposition:")
    print(f"  Uncertainty:  {uncertainty:.4f}  [Inherent data randomness]")
    print(f"  Resolution:   {resolution:.4f}  [How well model separates classes]")
    print(f"  Reliability:  {reliability:.4f}  [Calibration error]")
    print(f"  ──────────────────────────")
    print(f"  Brier Score:  {reliability - resolution + uncertainty:.4f}\\n")

    # 1.3.3 Confidence vs Accuracy Analysis
    print("1.3.3 Confidence-Stratified Performance")
    print("-" * 80 + "\\n")

    # Stratify predictions by confidence level
    confidence_levels = [
        ('Very Low', 0.5, 0.6),
        ('Low', 0.6, 0.7),
        ('Medium', 0.7, 0.8),
        ('High', 0.8, 0.9),
        ('Very High', 0.9, 1.0)
    ]

    confidence_analysis = []

    for level_name, low, high in confidence_levels:
        # Get maximum probability (confidence) for each prediction
        max_probs = np.maximum(y_proba_calib, 1 - y_proba_calib)
        mask = (max_probs >= low) & (max_probs < high)

        if level_name == 'Very High':  # Include 1.0 in last bin
            mask = mask | (max_probs == 1.0)

        n_samples = mask.sum()
        if n_samples == 0:
            continue

        y_pred_conf = (y_proba_calib[mask] >= 0.5).astype(int)
        acc = accuracy_score(y_test[mask], y_pred_conf)
        avg_conf = max_probs[mask].mean()

        confidence_analysis.append({
            'Confidence_Level': level_name,
            'Range': f'[{low:.1f}, {high:.1f})',
            'n_samples': int(n_samples),
            'Avg_Confidence': avg_conf,
            'Accuracy': acc,
            'Gap': acc - avg_conf
        })

    df_confidence = pd.DataFrame(confidence_analysis)

    print("Performance by Confidence Level:\\n")
    print(df_confidence.to_string(index=False, float_format='%.4f'))
    print()

    df_confidence.to_csv(f'{OUTPUT_PATH}/tier1_confidence_stratified.csv', index=False)
    print(f"✓ Confidence analysis saved\\n")

    # 1.3.4 Visualization
    print("1.3.4 Creating visualizations...")
    print()

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Calibration curve
    ax1 = fig.add_subplot(gs[0, :2])

    ax1.plot(prob_pred, prob_true, 's-', linewidth=2.5, markersize=8,
            color='#3498db', label='Model Calibration')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.5)

    # Shade area for calibration error
    ax1.fill_between(prob_pred, prob_true, prob_pred, alpha=0.3, color='red',
                    label=f'Calibration Error (ECE={ece:.4f})')

    ax1.set_xlabel('Mean Predicted Probability', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Fraction of Positives (Accuracy)', fontweight='bold', fontsize=12)
    ax1.set_title('Calibration Curve (Reliability Diagram)', fontweight='bold', fontsize=14, pad=10)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Plot 2: Histogram of probabilities
    ax2 = fig.add_subplot(gs[0, 2])

    ax2.hist(y_proba_calib, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Predicted Probability', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Frequency', fontweight='bold', fontsize=12)
    ax2.set_title('Prediction Confidence Distribution', fontweight='bold', fontsize=13, pad=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Decision Threshold')
    ax2.legend()

    # Plot 3: Calibration error by bin
    ax3 = fig.add_subplot(gs[1, 0])

    colors = ['#2ecc71' if err < 0.05 else '#f39c12' if err < 0.15 else '#e74c3c'
             for err in df_ece['Calibration_Error']]

    bars = ax3.bar(range(len(df_ece)), df_ece['Calibration_Error'], color=colors,
                  alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Confidence Bin', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Calibration Error', fontweight='bold', fontsize=12)
    ax3.set_title('Calibration Error by Bin', fontweight='bold', fontsize=13, pad=10)
    ax3.set_xticks(range(len(df_ece)))
    ax3.set_xticklabels([f"{i+1}" for i in range(len(df_ece))], fontsize=10)
    ax3.axhline(y=0.05, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Good (<0.05)')
    ax3.axhline(y=0.15, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Fair (<0.15)')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Confidence vs Accuracy
    ax4 = fig.add_subplot(gs[1, 1])

    ax4.plot(df_confidence['Avg_Confidence'], df_confidence['Accuracy'], 'o-',
            linewidth=2.5, markersize=10, color='#3498db')
    ax4.plot([0.5, 1], [0.5, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect Calibration')

    for i, row in df_confidence.iterrows():
        ax4.annotate(row['Confidence_Level'],
                    (row['Avg_Confidence'], row['Accuracy']),
                    textcoords="offset points", xytext=(0,10), ha='center',
                    fontsize=9, fontweight='bold')

    ax4.set_xlabel('Average Confidence', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    ax4.set_title('Confidence vs Accuracy', fontweight='bold', fontsize=13, pad=10)
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_xlim([0.5, 1])
    ax4.set_ylim([0.5, 1])

    # Plot 5: Sample distribution by confidence
    ax5 = fig.add_subplot(gs[1, 2])

    colors_conf = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60']
    bars = ax5.barh(range(len(df_confidence)), df_confidence['n_samples'],
                   color=colors_conf[:len(df_confidence)], alpha=0.7, edgecolor='black')
    ax5.set_yticks(range(len(df_confidence)))
    ax5.set_yticklabels(df_confidence['Confidence_Level'])
    ax5.set_xlabel('Number of Predictions', fontweight='bold', fontsize=12)
    ax5.set_title('Sample Distribution by Confidence', fontweight='bold', fontsize=13, pad=10)
    ax5.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_confidence['n_samples'])):
        ax5.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val}', va='center', fontweight='bold')

    plt.suptitle('Confidence Calibration Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(f'{OUTPUT_PATH}/tier1_confidence_calibration.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ Figure saved: tier1_confidence_calibration.png\\n")

    # 1.3.5 Key Insights
    print("-" * 80)
    print("KEY INSIGHTS")
    print("-" * 80 + "\\n")

    insights_calib = []

    if ece < 0.05:
        insights_calib.append(f"• ECE = {ece:.4f}: EXCELLENT calibration")
        insights_calib.append("  → Predicted probabilities reliably reflect true accuracy")
    elif ece < 0.15:
        insights_calib.append(f"• ECE = {ece:.4f}: MODERATE calibration")
        insights_calib.append("  → Some miscalibration present, consider calibration methods")
    else:
        insights_calib.append(f"• ECE = {ece:.4f}: POOR calibration")
        insights_calib.append("  → Predicted probabilities don't reflect true confidence")
        insights_calib.append("  → Strongly recommend Platt scaling or isotonic regression")

    insights_calib.append(f"\\n• Brier Score = {brier:.4f}")
    if brier < 0.1:
        insights_calib.append("  → Excellent probabilistic predictions")
    elif brier < 0.2:
        insights_calib.append("  → Good probabilistic predictions")
    else:
        insights_calib.append("  → Room for improvement in probability estimates")

    # Check if model is over/under-confident
    avg_gap = df_confidence['Gap'].mean()
    if avg_gap > 0.05:
        insights_calib.append("\\n• Model is UNDER-CONFIDENT")
        insights_calib.append("  → Actual accuracy exceeds predicted confidence")
        insights_calib.append("  → Safe but may underutilize high-quality predictions")
    elif avg_gap < -0.05:
        insights_calib.append("\\n• Model is OVER-CONFIDENT")
        insights_calib.append("  → Predicted confidence exceeds actual accuracy")
        insights_calib.append("  → Risky - may trust incorrect predictions")
    else:
        insights_calib.append("\\n• Model is WELL-CALIBRATED")
        insights_calib.append("  → Confidence matches accuracy across levels")

    for insight in insights_calib:
        print(insight)

    print()

print("="*80)
print("TIER 1.3 COMPLETE")
print("="*80)"""
    }
]

# Add cells to notebook
for tier in remaining_tiers:
    cell = {
        "cell_type": "code",
        "source": tier["source"],
        "metadata": {},
        "execution_count": None,
        "outputs": []
    }
    notebook['cells'].append(cell)

# Save notebook
with open('MAINTAINABILITY_PREDICTION.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"✓ Added {len(remaining_tiers)} tier cell(s) to notebook")
print(f"✓ Total cells now: {len(notebook['cells'])}")
