import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    # Read the CSV and add data to a list of dictionaries
    data = []
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data.append({
                "Question": row["Question - English"],
                "Answer": row["Answer - English"]
            })
    
    # Write the list of dictionaries to a JSON file
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

# Specify the input CSV file and output JSON file paths
csv_file_path = 'input.csv'
json_file_path = 'qa.json'

# Convert the file
csv_to_json(csv_file_path, json_file_path)

print(f"Conversion completed. JSON file saved to: {json_file_path}")
