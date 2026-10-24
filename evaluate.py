
import json
from main import verify_claim


# ----------------------------------
# 1. LOAD TEST CASES
# ----------------------------------

with open("evaluation_cases.json", "r", encoding="utf-8") as file:
    test_cases = json.load(file)


# ----------------------------------
# 2. RUN EVALUATION
# ----------------------------------

correct = 0
incorrect = 0
results = []

for index, test in enumerate(test_cases, start=1):

    claim = test["claim"]
    evidence = test["evidence"]
    expected = test["expected"].strip().upper()

    try:
        result = verify_claim(
            claim,
            None,
            evidence
        )

        actual = result["classification"].strip().upper()
        explanation = result["explanation"]

    except Exception as error:
        actual = "VERIFICATION_ERROR"
        explanation = str(error)

    is_correct = actual == expected

    if is_correct:
        correct += 1
    else:
        incorrect += 1

    results.append({
        "test_case": index,
        "claim": claim,
        "evidence": evidence,
        "expected": expected,
        "actual": actual,
        "correct": is_correct,
        "explanation": explanation
    })

    print("\n" + "=" * 60)
    print("TEST CASE:", index)
    print("Claim:", claim)
    print("Expected:", expected)
    print("Actual:", actual)
    print("Correct:", is_correct)
    print("Explanation:", explanation)


# ----------------------------------
# 3. CONFUSION MATRIX
# ----------------------------------

labels = [
    "SUPPORTED",
    "CONTRADICTED",
    "NOT_SUPPORTED",
    "INSUFFICIENT_EVIDENCE"
]

# Rows = Expected, Columns = Predicted
matrix = [[0 for _ in labels] for _ in labels]

for result in results:
    expected = result["expected"]
    actual = result["actual"]

    # Only count recognized classification labels
    if expected in labels and actual in labels:
        row = labels.index(expected)
        col = labels.index(actual)
        matrix[row][col] += 1

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("Rows = Expected | Columns = Predicted\n")

print(f"{'EXPECTED / PREDICTED':<25}", end="")
for label in labels:
    print(f"{label[:12]:>15}", end="")
print()

for index, row_values in enumerate(matrix):
    print(f"{labels[index]:<25}", end="")
    for value in row_values:
        print(f"{value:>15}", end="")
    print()


# ----------------------------------
# 4. PRECISION, RECALL, F1-SCORE
# ----------------------------------

print("\n" + "=" * 75)
print("PER-CLASS METRICS")
print("=" * 75)

print(
    f"{'CLASS':<25}"
    f"{'PRECISION':>12}"
    f"{'RECALL':>12}"
    f"{'F1-SCORE':>12}"
    f"{'SUPPORT':>12}"
)

for i, label in enumerate(labels):

    # True Positives: correct predictions for this class
    tp = matrix[i][i]

    # False Positives: other actual classes predicted as this class
    fp = sum(matrix[row][i] for row in range(len(labels))) - tp

    # False Negatives: this actual class predicted as other classes
    fn = sum(matrix[i]) - tp

    # Number of actual examples belonging to this class
    support = sum(matrix[i])

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    print(
        f"{label:<25}"
        f"{precision:>12.3f}"
        f"{recall:>12.3f}"
        f"{f1_score:>12.3f}"
        f"{support:>12}"
    )


# ----------------------------------
# 5. OVERALL SUMMARY
# ----------------------------------

total = len(results)
accuracy = correct / total if total else 0.0

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("Total test cases:", total)
print("Correct predictions:", correct)
print("Incorrect predictions:", incorrect)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("=" * 60)


# ----------------------------------
# 6. SAVE RESULTS
# ----------------------------------

with open(
    "evaluation_results.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        results,
        file,
        indent=2,
        ensure_ascii=False
    )

print("Results saved to evaluation_results.json")
