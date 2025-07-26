import csv
import anthropic
import json
import os
import time

# --- Configuration ---

# IMPORTANT: Set your Anthropic API key as an environment variable for security.
# In your terminal (Linux/macOS): export ANTHROPIC_API_KEY='YOUR_API_KEY'
# In PowerShell (Windows): $env:ANTHROPIC_API_KEY='YOUR_API_KEY'
# The Anthropic client library automatically uses this environment variable.
try:
    client = anthropic.Anthropic()
except anthropic.APIKeyNotFoundError as e:
    print("FATAL ERROR: ANTHROPIC_API_KEY environment variable not set.")
    print("Please set the environment variable and try again.")
    client = None # Set client to None to prevent further execution

API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-20250514" # Or another suitable Claude model

# File paths
INPUT_CSV_FILE = '/home/bowang/Documents/alif/clinical-camel-asr/data_generation/uhn8_input.csv'
OUTPUT_CSV_FILE = '/home/bowang/Documents/alif/clinical-camel-asr/data_generation/generated_uhn8_dialogues_claude2.csv'
PROMPT_FILE = '/home/bowang/Documents/alif/clinical-camel-asr/prompts/transcript_from_json_v2.txt' # The file containing the prompt template

# Column name in the input CSV containing the JSON data
JSON_COLUMN_NAME = 'SOAP-NOTE-4-fileds'


def generate_dialogue_from_json(json_summary: str, prompt_template: str) -> str:
    """
    Sends a request to the Anthropic Claude API to generate a medical dialogue.

    Args:
        json_summary: A string containing the JSON data for a medical encounter.
        prompt_template: The string template for the prompt.

    Returns:
        The generated dialogue as a string, or an error message if the API call fails.
    """
    if not client:
        return "ERROR: Anthropic client not initialized due to missing API key."

    # Replace the placeholder in the template with the actual JSON data
    full_prompt = prompt_template.replace('${input}', json_summary)

    try:
        # Make the API request using the anthropic library
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
        )
        
        # Extract the generated text from the response object
        if message.content:
            return message.content[0].text
        else:
            return f"ERROR: API call was successful but returned no content. Full response: {message}"

    except anthropic.APIError as e:
        return f"ERROR: An Anthropic API error occurred: {e}"
    except Exception as e:
        return f"ERROR: An unexpected error occurred: {e}"


def process_csv():
    """
    Reads the input CSV, generates dialogues for each row, and saves to an output CSV.
    """
    if not client:
        return # Stop execution if the client wasn't initialized

    print("Starting to process...")

    # --- Read the prompt template from the file ---
    try:
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
            prompt_template_content = f.read()
        print(f"Successfully loaded prompt template from '{PROMPT_FILE}'.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Prompt template file not found at '{PROMPT_FILE}'.")
        print("Please make sure the prompt file exists in the same directory as the script.")
        return
    except Exception as e:
        print(f"FATAL ERROR: Could not read the prompt file: {e}")
        return

    # --- Check for input CSV file ---
    if not os.path.exists(INPUT_CSV_FILE):
        print(f"Error: Input file not found at '{INPUT_CSV_FILE}'")
        return

    # --- Process the CSV file ---
    try:
        with open(INPUT_CSV_FILE, mode='r', encoding='utf-8') as infile, \
             open(OUTPUT_CSV_FILE, mode='w', encoding='utf-8', newline='') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # Write the header for the output file
            writer.writerow(['Generated Transcript', 'Original JSON'])

            # Get the header row from the input file and find the column index
            header = next(reader)
            try:
                json_column_index = header.index(JSON_COLUMN_NAME)
            except ValueError:
                print(f"Error: Column '{JSON_COLUMN_NAME}' not found in the input CSV.")
                print(f"Available columns are: {header}")
                return
            
            # Process each row in the CSV
            for i, row in enumerate(reader):
                # Ensure the row has enough columns to avoid index errors
                if len(row) > json_column_index:
                    json_data_str = row[json_column_index]

                    if json_data_str and json_data_str.strip():
                        print(f"Processing row {i+2}...")
                        # Pass the loaded prompt template to the generation function
                        generated_transcript = generate_dialogue_from_json(json_data_str, prompt_template_content)
                        
                        # Write the result to the new CSV
                        writer.writerow([generated_transcript, json_data_str])
                        
                        # A small delay to respect API rate limits, if any.
                        time.sleep(1) 
                    else:
                        print(f"Skipping row {i+2} due to empty JSON data.")
                        writer.writerow(['SKIPPED - Empty JSON', ''])
                else:
                    print(f"Skipping row {i+2} as it does not have the required column.")

    except FileNotFoundError:
        print(f"Error: Could not find the input file: {INPUT_CSV_FILE}")
    except Exception as e:
        print(f"An unexpected error occurred during file processing: {e}")

    print(f"\nProcessing complete. Results saved to {OUTPUT_CSV_FILE}")


if __name__ == "__main__":
    # This block ensures the script runs when executed directly
    process_csv()
