import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    # Read the CSV and convert it to a list of dictionaries
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = [row for row in csv_reader]
    
    # Write the data to a JSON file
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

# Example usage
csv_file_path = 'oncology_qa.csv'  # Replace with your CSV file path
json_file_path = 'oncology_qa.json'  # Replace with your desired JSON file path

csv_to_json(csv_file_path, json_file_path)
