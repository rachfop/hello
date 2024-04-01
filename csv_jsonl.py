import csv
import json


# Open the CSV file
def csv_jsonl():
    with open("input.csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file)

        # Open the JSONL file for writing
        with open("output.jsonl", "w") as jsonl_file:
            # Skip the header row
            next(csv_reader)

            # Iterate over each row in the CSV
            for row in csv_reader:
                # Extract the prompt and response from the row
                prompt = row[2]  # Assuming the prompt is in the third column (index 2)
                response = row[
                    3
                ]  # Assuming the response is in the fourth column (index 3)

                # Create a dictionary with the prompt and response
                data = {"prompt": prompt, "response": response}

                # Convert the dictionary to JSON and write it to the JSONL file
                json_data = json.dumps(data)
                jsonl_file.write(json_data + "\n")


csv_jsonl()
