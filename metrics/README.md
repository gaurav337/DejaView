# Metrics & Evaluation

This directory contains scripts to generate test data and evaluate the performance (F1 Score, Precision, Recall) of the Near-Duplicate Image Detection system.

## Scripts

### 1. `create_evaluation_data.py`
Generates transformed versions of existing images (resize, crop, compress, color, watermark).
- **Output**: `evaluation/test_images/should_match/`
- **Generates**: `evaluation/ground_truth_matches.csv`

### 2. `add_non_matching.py`
Generates synthetic random images (noise, solid colors) that should NOT match any existing image.
- **Output**: `evaluation/test_images/should_not_match/`
- **Generates**: `evaluation/ground_truth_non_matches.csv`

### 3. `combine_ground_truth.py`
Combines the positive (matches) and negative (non-matches) ground truth files into one master file.
- **Output**: `evaluation/ground_truth.csv`

### 4. `evaluate.py`
Runs the detection pipeline against the ground truth dataset and calculates metrics.
- **Input**: `evaluation/ground_truth.csv`
- **Output**: 
    - Console Output: F1 Score, Confusion Matrix, Per-transform accuracy.
    - File Output: `evaluation/detailed_results.csv`

## How to Run

Run the scripts in the following order from the project root (`d:\clipupdated\DejaView`):

```bash
# 1. Generate Positive Test Cases (Transformed Images)
python metrics/create_evaluation_data.py

# 2. Generate Negative Test Cases (Unique/Random Images)
python metrics/add_non_matching.py

# 3. Combine Ground Truth Data
python metrics/combine_ground_truth.py

# 4. Run Evaluation
python metrics/evaluate.py
```
