import json
import csv
import random
import re

def main():
    json_file = "fib_ar.json"  # Path to your dataset
    csv_output = "random_predictions.csv"  # Where to store predictions

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    letters = ["أ", "ب", "ج", "د"]
    predictions = []

    for entry in data:
        question_text = entry.get("Question", "")
        full_answer = entry.get("Answer", "").strip()

        # 1) Extract ONLY the first letter from the set [أ, ب, ج, د]
        #    e.g. if "Answer" is "ج. خلايا..." this picks out "ج".
        match = re.search(r"[أبجد]", full_answer)
        if match:
            correct_letter = match.group(0)  # e.g. 'ج'
        else:
            correct_letter = ""  # fallback if no letter found

        # 2) Generate a random guess
        random_guess = random.choice(letters)

        predictions.append({
            "Question": question_text,
            "CorrectAnswer": correct_letter,
            "RandomGuess": random_guess
        })

    # 3) Write to CSV
    with open(csv_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Question", "CorrectAnswer", "RandomGuess"])
        writer.writeheader()
        writer.writerows(predictions)

    # 4) Evaluate accuracy
    total = len(predictions)
    correct_count = sum(1 for p in predictions if p["RandomGuess"] == p["CorrectAnswer"])
    accuracy = correct_count / total if total > 0 else 0.0

    print(f"Random predictions saved to: {csv_output}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
