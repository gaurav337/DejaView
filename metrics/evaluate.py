import csv
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

try:
    from duplicate_checker import check_image_pipeline
except ImportError as e:
    print(f"Error importing duplicate_checker: {e}")
    print(f"Make sure you are running this script from the correct environment and {PARENT_DIR} contains duplicate_checker.py")
    exit(1)

EVAL_DIR = os.path.join(CURRENT_DIR, "evaluation")
GROUND_TRUTH_PATH = os.path.join(EVAL_DIR, "ground_truth.csv")
DETAILED_RESULTS_PATH = os.path.join(EVAL_DIR, "detailed_results.csv")

def run_evaluation():
    
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"Error: Ground truth file not found at {GROUND_TRUTH_PATH}")
        print("Please run combine_ground_truth.py first.")
        return

    results = {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
    }
    
    transform_results = {}

    detailed_results = []
    
    print("=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70)
    
    with open(GROUND_TRUTH_PATH, "r") as f:
        reader = csv.DictReader(f)
        total = 0
        
        for row in reader:
            total += 1
            test_image = row["test_image"]
            expected = int(row["expected"])
            transform = row["transform"]
            
            if not os.path.exists(test_image):
                 print(f"  Warning: Test image not found: {test_image}. Skipping.")
                 continue

            try:
                result = check_image_pipeline(test_image)
                
                if result["status"] in ["Similar", "Duplicate"]:
                    predicted = 1
                else:
                    predicted = 0
                    
                method = result.get("method", "N/A")
                similarity = result.get("similarity_percentage", 0)
                
            except Exception as e:
                print(f"  ERROR processing {test_image}: {e}")
                predicted = 0
                method = "error"
                similarity = 0
            
            if expected == 1 and predicted == 1:
                results["TP"] += 1
                status = "[TP]"
            elif expected == 0 and predicted == 0:
                results["TN"] += 1
                status = "[TN]"
            elif expected == 0 and predicted == 1:
                results["FP"] += 1
                status = "[FP]"
            else:
                results["FN"] += 1
                status = "[FN]"
            
            if transform not in transform_results:
                transform_results[transform] = {"correct": 0, "total": 0}
            transform_results[transform]["total"] += 1
            if (expected == predicted):
                transform_results[transform]["correct"] += 1
            
            detailed_results.append({
                "test_image": test_image,
                "expected": expected,
                "predicted": predicted,
                "status": status,
                "method": method,
                "similarity": similarity
            })
            
            print(f"[{total}] {status} | {transform:12} | Expected:{expected} Got:{predicted} | {method} | {similarity:.1f}%")
    
    print("\n" + "=" * 70)
    
    TP, TN, FP, FN = results["TP"], results["TN"], results["FP"], results["FN"]
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    
    print("\nCONFUSION MATRIX")
    print("-" * 40)
    print(f"                  Predicted")
    print(f"                  Dup    Unique")
    print(f"Actual Duplicate   {TP:<6} {FN:<6}  (TP, FN)")
    print(f"Actual Unique      {FP:<6} {TN:<6}  (FP, TN)")
    
    print("\nMETRICS")
    print("-" * 40)
    print(f"  True Positives (TP):  {TP}")
    print(f"  True Negatives (TN):  {TN}")
    print(f"  False Positives (FP): {FP}")
    print(f"  False Negatives (FN): {FN}")
    print()
    print(f"  Accuracy:   {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision:  {precision:.4f}  ({precision*100:.2f}%)")
    print(f"  Recall:     {recall:.4f}  ({recall*100:.2f}%)")
    print(f"  +-------------------------------+")
    print(f"  |  F1 SCORE:  {f1:.4f}           |")
    print(f"  +-------------------------------+")
    
    print("\nPER-TRANSFORMATION ACCURACY")
    print("-" * 40)
    for transform, data in sorted(transform_results.items()):
        acc = data["correct"] / data["total"] if data["total"] > 0 else 0
        bar = "#" * int(acc * 20) + "." * (20 - int(acc * 20))
        print(f"  {transform:12} |{bar}| {acc*100:5.1f}% ({data['correct']}/{data['total']})")
    
    with open(DETAILED_RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["test_image", "expected", "predicted", "status", "method", "similarity"])
        writer.writeheader()
        writer.writerows(detailed_results)
    
    print(f"\n[SAVED] Detailed results saved to: {DETAILED_RESULTS_PATH}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": results,
        "per_transform": transform_results
    }


if __name__ == "__main__":
    results = run_evaluation()
