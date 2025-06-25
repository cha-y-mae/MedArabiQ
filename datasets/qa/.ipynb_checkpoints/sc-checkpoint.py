import csv
import json
import math

def clean_nans(row):
    # Replace NaN/empty values with None (valid in JSON)
    return {
        key: (None if value == '' or value is None or (isinstance(value, float) and math.isnan(value)) else value)
        for key, value in row.items()
    }

def csv_to_json(csv_file_path, json_file_path):
    data = []
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data.append(clean_nans(row))  # clean each row

    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

# File paths
csv_file_path = 'data.csv'
json_file_path = 'output.json'

csv_to_json(csv_file_path, json_file_path)
print(f"saved to {json_file_path}")
