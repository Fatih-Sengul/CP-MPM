import json

with open('/home/user/CP-MPM/MAINTAINABILITY_PREDICTION.ipynb', 'r') as f:
    nb = json.load(f)

print("CREATING CLEAN NOTEBOOK...")
print("="*80)

# Clean Cell 0: Setup only
cell_0 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": """\"\"\"
================================================================================
SETUP: Reproducibility and Helper Functions
================================================================================
Run this cell FIRST.
================================================================================
\"\"\"

import numpy as np
import random
import os
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

print("✓ Random seeds set (seed=42)")

# Helper functions
def save_figure(fig, filename, dpi=300):
    if 'FIGURES_PATH' in globals():
        filepath = os.path.join(FIGURES_PATH, filename)
        fig.tight_layout()
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
    else:
        print("⚠ FIGURES_PATH not defined")

def print_classification_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"Confusion Matrix:\\n{confusion_matrix(y_true, y_pred)}")

def calculate_expected_calibration_error(y_true, y_proba, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bin_edges[1:-1])
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_proba[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * abs(bin_acc - bin_conf)
    return ece

print("=" * 80)
print("SETUP COMPLETE")
print("=" * 80)
""".split('\n')
}

# Clean Cell 1: ALL imports + Drive + Paths
cell_1 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": """\"\"\"
================================================================================
SECTION 1: IMPORTS, DRIVE MOUNT, AND PATH SETUP
================================================================================
\"\"\"

# =============================================================================
# MOUNT GOOGLE DRIVE
# =============================================================================

from google.colab import drive
drive.mount('/content/drive')

print("✓ Google Drive mounted")

# =============================================================================
# ALL IMPORTS (ORGANIZED)
# =============================================================================

# Core libraries
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

# Static analysis tools
import lizard
import javalang

# ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, balanced_accuracy_score,
    roc_curve, precision_recall_curve
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

print("✓ All libraries imported")

# =============================================================================
# PATH DEFINITIONS
# =============================================================================

BASE_PATH = '/content/drive/MyDrive/ieee'
LABELS_PATH = f'{BASE_PATH}/labels.csv'
SOURCE_PATH = f'{BASE_PATH}/dataset_source_files'
OUTPUT_PATH = f'{BASE_PATH}/static_analysis_results'
FIGURES_PATH = f'{BASE_PATH}/figures'

# Create output directories
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

print("\\n" + "="*80)
print("PATHS CONFIGURED")
print("="*80)
print(f"Labels:  {LABELS_PATH}")
print(f"Source:  {SOURCE_PATH}")
print(f"Output:  {OUTPUT_PATH}")
print(f"Figures: {FIGURES_PATH}")
print("="*80)
""".split('\n')
}

print("✓ Created clean Cell 0 and Cell 1")

# Update notebook
nb['cells'][0] = cell_0
nb['cells'][1] = cell_1

# Save
with open('/home/user/CP-MPM/MAINTAINABILITY_PREDICTION_CLEAN.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("✓ Clean notebook saved to MAINTAINABILITY_PREDICTION_CLEAN.ipynb")
