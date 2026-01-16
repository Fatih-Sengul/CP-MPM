# 📘 MAINTAINABILITY PREDICTION NOTEBOOK - USAGE GUIDE

## ⚠️ IMPORTANT: Read This First!

This notebook **MUST** be run in **exact sequential order**. Skipping cells will cause errors!

---

## 📋 CELL-BY-CELL GUIDE

### **Cell 0: Setup** ✅
**What it does:**
- Sets random seeds for reproducibility (seed=42)
- Defines helper functions: `save_figure()`, `print_classification_metrics()`, `calculate_expected_calibration_error()`

**Expected Output:**
```
✓ Random seeds set (seed=42)
================================================================================
SETUP COMPLETE
================================================================================
```

**If this fails:** Check that numpy and random are available

---

### **Cell 1: Data Loading + Metric Extraction** 📊
**What it does:**
- Mounts Google Drive
- Installs packages (lizard, javalang, shap)
- Imports ALL libraries
- Sets up paths (BASE_PATH, LABELS_PATH, etc.)
- Loads labels.csv
- Extracts static code metrics from Java files
- Saves extracted_metrics.csv

**Expected Output:**
```
Drive already mounted at /content/drive
✓ All libraries imported
✓ Loaded 304 samples
Processing 0/304...
Processing 50/304...
...
✓ Successfully extracted metrics for ~231/304 files
✗ Failed: ~73 files
```

**Runtime:** ~3-5 minutes (metric extraction is slow)

**If errors occur:**
- "File not found": Check that `/content/drive/MyDrive/ieee/` exists
- "73 files failed": **This is normal!** (case-sensitive paths, missing files)

**Variables created:**
- `df` = DataFrame with labels
- `df_metrics` = DataFrame with extracted metrics
- `BASE_PATH`, `FIGURES_PATH`, etc.

---

### **Cell 2: ML Modeling** 🤖
**What it does:**
- Defines 17 features (nloc, ccn, token_count, ...)
- Creates `feature_cols` and `feature_names` (IMPORTANT!)
- Splits data: train/test (80/20)
- Scales features with StandardScaler
- Trains 3 models: Logistic Regression, Random Forest, Gradient Boosting
- Performs 5-fold cross-validation
- Saves best model as `best_model`
- SHAP explainability analysis

**Expected Output:**
```
Dataset: 231 samples, 17 features
Features: nloc, ccn, token_count, ...

Data Split:
  Training: 184 samples
  Test: 47 samples

Logistic Regression  : CV = 0.8XXX (±0.0XX), Test = 0.8XXX
Random Forest        : CV = 0.8XXX (±0.0XX), Test = 0.8XXX
Gradient Boosting    : CV = 0.8XXX (±0.0XX), Test = 0.8XXX

Best model: Random Forest (Test Accuracy: 0.8XXX)
```

**Runtime:** ~1-2 minutes

**Variables created (CRITICAL FOR NEXT CELLS):**
- `X_train_scaled`, `X_test_scaled`
- `y_train`, `y_test`
- `feature_cols`, `feature_names`
- `best_model`
- `df_results`

**⚠️ DO NOT SKIP THIS CELL!** Cell 3 depends on these variables!

---

### **Cell 3: Baseline Comparison** 📊
**What it does:**
- **DEPENDENCY CHECK**: Verifies Cell 2 was run
- Trains 5 baseline models:
  1. Majority Class (always predicts most frequent)
  2. Stratified Random (chance-level performance)
  3. Single-Feature (token_count only)
  4. Simple Decision Tree (max_depth=3)
  5. Logistic Regression (simple)
- Compares with best model from Cell 2
- Creates visualization: `02_baseline_comparison.png`

**Expected Output:**
```
✓ All required variables found from Cell 2
  - X_train_scaled: (184, 17)
  - y_train: 184 samples
  - feature_cols: 17 features

1. Majority Class Baseline
  Accuracy:          0.7447

2. Stratified Random Baseline
  Accuracy:          0.6809

...

BASELINE COMPARISON SUMMARY
  Our Best Model (Random Forest)  0.8XXX
  Simple Decision Tree             0.7XXX
  Majority Class                   0.7447

✓ Improvement over best baseline: +XX.XX%
✓ Saved: 02_baseline_comparison.png
```

**Runtime:** ~30 seconds

**If you get NameError:**
```
❌ ERROR: MISSING REQUIRED VARIABLES
Missing: ['X_train_scaled', 'X_test_scaled', 'y_train', 'y_test']

⚠️  YOU MUST RUN CELL 2 (SECTION 2: ML MODELING) FIRST!
```
→ **Solution:** Go back and run Cell 2!

---

### **Cell 4: Robustness Analysis** 🔬
**What it does:**
- Bootstrap confidence intervals
- Token count dominance analysis

**Runtime:** ~1 minute

---

### **Cell 5: LOPO Cross-Validation** 🔄
**What it does:**
- Leave-One-Project-Out validation
- Per-project accuracy analysis
- Heatmap visualization

**Expected Output:**
```
LOPO Average Accuracy: 0.86 (±0.098)

Project Performance:
  JUnit4:          95.4%
  ArgoUML:         89.6%
  AOI:             84.0%
  DiaryManagement: 72.7%
```

**Runtime:** ~2-3 minutes

---

### **Cell 6: Expert Consensus Analysis** 👥
**What it does:**
- Extracts expert confidence from EM probabilities
- Creates confidence bins (Low/Medium/High consensus)
- Evaluates model performance per bin
- Saves: `04_expert_confidence_distribution.png`, `04_disagreement_vs_performance.png`

**Runtime:** ~1 minute

---

### **Cells 7-11: TIER Analyses** 🎯
- TIER 1.1: Error Analysis
- TIER 1.2: Threshold Optimization
- TIER 1.3: Confidence Calibration
- TIER 2: Advanced Metrics
- TIER 3: Hyperparameter Tuning

**Runtime:** ~5-10 minutes total

---

### **Cell 12: Final Summary** 📝
**What it does:**
- Markdown summary of findings, limitations, future work

**Runtime:** Instant (markdown cell)

---

## 🚀 QUICK START CHECKLIST

```
□ Open notebook in Google Colab
□ Runtime → Change runtime type → GPU (optional, helps with SHAP)
□ Run Cell 0 (Setup) ✅
□ Run Cell 1 (Data Loading) ✅ - Wait ~3-5 min
□ Run Cell 2 (ML Modeling) ✅ - Wait ~1-2 min
□ Run Cell 3 (Baseline) ✅
□ Run Cells 4-12 sequentially ✅
□ Check /content/drive/MyDrive/ieee/figures/ for saved plots
```

---

## 🔧 TROUBLESHOOTING

### "NameError: name 'X_train_scaled' is not defined"
**Cause:** You skipped Cell 2
**Solution:** Run Cell 2 first!

### "NameError: name 'feature_cols' is not defined"
**Cause:** Cell 2 didn't complete successfully
**Solution:** Re-run Cell 2 and wait for it to finish

### "73 files failed extraction"
**Cause:** Case-sensitive paths (jsweet vs Jsweet), missing files
**Solution:** **This is expected!** 231/304 files is sufficient

### "SyntaxError: invalid syntax"
**Cause:** Corrupted cell or Python version mismatch
**Solution:** Runtime → Restart runtime, run from Cell 0

### Figures not saving
**Cause:** `FIGURES_PATH` not defined or Drive not mounted
**Solution:** Re-run Cell 1 (sets up FIGURES_PATH)

---

## 📊 EXPECTED RESULTS

| Metric | Value |
|--------|-------|
| **Dataset Size** | 231 classes (after extraction) |
| **Features** | 17 static code metrics |
| **Train/Test Split** | 184 / 47 |
| **Best Model** | Random Forest |
| **Test Accuracy** | ~85-90% |
| **LOPO Accuracy** | ~86% (±9.8%) |
| **Baseline Improvement** | +12-15% over majority class |

---

## 📁 OUTPUT FILES

After running all cells, check these locations:

```
/content/drive/MyDrive/ieee/
├── static_analysis_results/
│   └── extracted_metrics.csv
└── figures/
    ├── 02_baseline_comparison.png
    ├── 04_expert_confidence_distribution.png
    ├── 04_disagreement_vs_performance.png
    ├── shap_summary_plot.png
    └── ... (other plots from TIER analyses)
```

---

## ⏱️ TOTAL RUNTIME

**Full notebook:** ~15-20 minutes

Breakdown:
- Cell 0: <1 sec
- Cell 1: 3-5 min (metric extraction)
- Cell 2: 1-2 min
- Cell 3: 30 sec
- Cells 4-11: 5-10 min
- Cell 12: Instant

---

## ✅ SUCCESS CRITERIA

You know it worked if:
1. ✅ Cell 1 extracts ~231 metrics
2. ✅ Cell 2 shows "Best model: Random Forest"
3. ✅ Cell 3 shows baseline comparison plot
4. ✅ Cell 5 shows LOPO accuracy ~86%
5. ✅ All figures saved to `/figures/` folder

---

## 📧 NEED HELP?

If something doesn't work:
1. Check this guide first
2. Read error messages carefully
3. Make sure you ran Cell 2 before Cell 3!
4. Try Runtime → Restart runtime and run all cells again

---

**Good luck!** 🚀
