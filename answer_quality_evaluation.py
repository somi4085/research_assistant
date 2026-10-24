
import json
import requests
from pathlib import Path

MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/generate"

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "answer_quality_cases.json"

VALID_LABELS = {
    "correctness": {"CORRECT", "INCORRECT"},
    "faithfulness": {"FAITHFUL", "UNFAITHFUL"},
    "relevance": {"RELEVANT", "IRRELEVANT"}
}


def evaluate_answer(question, evidence, answer):
    prompt = f"""
You are a strict evaluator for a Retrieval-Augmented Generation system.

Evaluate the answer using ONLY the question and evidence provided.

QUESTION:
{question}

EVIDENCE:
{evidence}

ANSWER:
{answer}

Evaluate three independent dimensions:

1. correctness:
   CORRECT = the answer's factual claims are correct based on the evidence.
   INCORRECT = at least one factual claim contradicts the evidence.

2. faithfulness:
   FAITHFUL = all factual claims in the answer are supported by the evidence.
   UNFAITHFUL = at least one factual claim is unsupported or contradicts the evidence.

3. relevance:
   RELEVANT = the answer attempts to address the user's question,
   even if its information is factually incorrect.

   IRRELEVANT = the answer does not address the question
   and instead gives unrelated information.

   Evaluate relevance independently from correctness and faithfulness.
   Do not label an answer IRRELEVANT merely because it is wrong
   or contradicts the evidence.
Do not assume facts that are not in the evidence.
Do not use the expected labels.
Return ONLY a valid JSON object with these exact keys:
{{
  "correctness": "CORRECT or INCORRECT",
  "faithfulness": "FAITHFUL or UNFAITHFUL",
  "relevance": "RELEVANT or IRRELEVANT"
}}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0}
        },
        timeout=120
    )

    response.raise_for_status()
    result = json.loads(response.json()["response"])

    for dimension, allowed_values in VALID_LABELS.items():
        label = result.get(dimension)

        if not isinstance(label, str):
            raise ValueError(f"Missing or invalid label: {dimension}")

        label = label.strip().upper()

        if label not in allowed_values:
            raise ValueError(
                f"Invalid {dimension} label returned by model: {label}"
            )

        result[dimension] = label

    return result


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    if not isinstance(test_cases, list) or not test_cases:
        raise ValueError("Dataset must be a non-empty JSON list.")

    dimensions = ["correctness", "faithfulness", "relevance"]
    correct_counts = {dimension: 0 for dimension in dimensions}
    results = []

    for index, test in enumerate(test_cases, start=1):
        print(f"\nEvaluating test case {index}/{len(test_cases)}...")

        try:
            predicted = evaluate_answer(
                test["question"],
                test["evidence"],
                test["answer"]
            )

            expected = test["expected"]
            case_result = {
                "test_case": index,
                "question": test["question"],
                "expected": expected,
                "predicted": predicted,
                "checks": {}
            }

            for dimension in dimensions:
                expected_label = expected[dimension].strip().upper()
                actual_label = predicted[dimension]

                is_correct = actual_label == expected_label
                case_result["checks"][dimension] = is_correct

                if is_correct:
                    correct_counts[dimension] += 1

                print(
                    f"{dimension}: expected={expected_label}, "
                    f"predicted={actual_label}, correct={is_correct}"
                )

            results.append(case_result)

        except Exception as error:
            print(f"Evaluation failed for test case {index}: {error}")
            results.append({
                "test_case": index,
                "error": str(error)
            })

    successful = [
        result for result in results
        if "checks" in result
    ]

    print("\n" + "=" * 55)
    print("ANSWER QUALITY EVALUATION SUMMARY")
    print("=" * 55)

    for dimension in dimensions:
        accuracy = (
            correct_counts[dimension] / len(successful) * 100
            if successful else 0
        )
        print(f"{dimension.title()} accuracy: {accuracy:.2f}%")

    print(f"Total test cases: {len(test_cases)}")
    print(f"Successfully evaluated: {len(successful)}")
    print(f"Failed evaluations: {len(test_cases) - len(successful)}")

    output_path = BASE_DIR / "answer_quality_results.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
