import csv
import json

def convert_csv_to_jsonl(input_csv_path, output_jsonl_path):
    """
    Reads a CSV file with 'raw_transcription' and 'ground_truth' columns
    and converts it into a JSON Lines file with a specific structure.

    Args:
        input_csv_path (str): The file path for the input CSV.
        output_jsonl_path (str): The file path for the output JSONL file.
    """
    print(f"Starting conversion from '{input_csv_path}' to '{output_jsonl_path}'...")
    
    try:
        # Open the input CSV file for reading and the output JSONL file for writing
        with open(input_csv_path, mode='r', encoding='utf-8') as csv_file, \
             open(output_jsonl_path, mode='w', encoding='utf-8') as jsonl_file:
            
            # Use DictReader to read the CSV as a list of dictionaries
            # This makes it easy to access columns by their header names
            csv_reader = csv.DictReader(csv_file)
            
            processed_rows = 0
            # Iterate over each row in the CSV file
            for row in csv_reader:
                # Ensure the required columns exist in the row
                if 'raw_transcription' not in row or 'ground_truth' not in row:
                    print(f"Warning: Skipping row {processed_rows + 1} due to missing columns.")
                    continue

                # Create the desired dictionary structure for the JSON output
                json_record = {
                    "inputs": {
                        "raw_transcription": row['raw_transcription']
                    },
                    "outputs": {
                        # As per your request, 'ground_truth' is mapped to 'cleaned_transcription'
                        "cleaned_transcription": row['ground_truth']
                    }
                }
                
                # Convert the Python dictionary to a JSON string
                # and write it to the output file, followed by a newline character.
                # This creates the JSON Lines (JSONL) format.
                jsonl_file.write(json.dumps(json_record) + '\n')
                processed_rows += 1

        print(f"Conversion successful. Processed {processed_rows} rows.")
        print(f"Output saved to '{output_jsonl_path}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- How to use the script ---
if __name__ == "__main__":
    # 1. Specify the path to your input CSV file.
    #    Make sure this file is in the same directory as the script,
    #    or provide the full path.
    input_file = 'promptim_wer.csv' 
    
    # 2. Specify the desired name for your output JSONL file.
    output_file = 'data.jsonl'
    
    # 3. Run the script.
    convert_csv_to_jsonl(input_file, output_file)

    # Example: To test, you can create a file named 'your_data.csv'
    # with the following content:
    #
    # audio_file,raw_transcription,ground_truth
    # "audio1.wav","um so like i was saying the the meeting is at 3pm","So, as I was saying, the meeting is at 3:00 PM."
    # "audio2.wav","can you repeat that please i didnt hear","Can you repeat that, please? I didn't hear.","
    # "audio3.wav","the quick brown fox jumps over the lazy dog","The quick brown fox jumps over the lazy dog."

