import csv
import json

def csv_to_json_with_ids(csv_file_path, json_file_path):
    """
    Convert a CSV file to a JSON file and add unique IDs to each row.

    Args:
        csv_file_path (str): Path to the input CSV file.
        json_file_path (str): Path to the output JSON file with IDs added.
    """
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = []
        for idx, row in enumerate(csv_reader, start=1):
            row['id'] = idx  # Add a unique ID to each row
            data.append(row)
    
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

# Example usage
csv_file_path = 'oncology_qa.csv'  # Replace with the path to your CSV file
json_file_path = 'oncology_qa.json'  # Desired output JSON file

csv_to_json_with_ids(csv_file_path, json_file_path)
