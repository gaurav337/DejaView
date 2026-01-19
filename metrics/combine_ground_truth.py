import csv
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(CURRENT_DIR, "evaluation")

matches = []
non_matches = []

matches_path = os.path.join(EVAL_DIR, "ground_truth_matches.csv")
non_matches_path = os.path.join(EVAL_DIR, "ground_truth_non_matches.csv")
output_path = os.path.join(EVAL_DIR, "ground_truth.csv")

if os.path.exists(matches_path):
    with open(matches_path, "r") as f:
        reader = csv.DictReader(f)
        matches = list(reader)
else:
    print(f"Warning: {matches_path} not found.")

if os.path.exists(non_matches_path):
    with open(non_matches_path, "r") as f:
        reader = csv.DictReader(f)
        non_matches = list(reader)
else:
    print(f"Warning: {non_matches_path} not found.")

all_data = matches + non_matches

if not all_data:
    print("No ground truth data found.")
    exit(1)

with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["original", "test_image", "transform", "expected"])
    writer.writeheader()
    writer.writerows(all_data)

print(f"Combined ground truth saved to: {output_path}")
print(f"  - Should match (duplicates): {len(matches)}")
print(f"  - Should NOT match (unique): {len(non_matches)}")
print(f"  - Total test cases: {len(all_data)}")
