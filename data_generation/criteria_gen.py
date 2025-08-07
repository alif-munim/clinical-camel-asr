import csv
import os
import json
import anthropic  # Make sure you have the 'anthropic' library installed: pip install anthropic

# --- Configuration ---
# IMPORTANT: Replace with your actual API key and file paths.
# It's highly recommended to use environment variables for your API key for security.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY")
PROMPT_TEMPLATE_FILE = '/home/bowang/Documents/alif/clinical-camel-asr/prompts/criteria_prompt_v7.txt'
INPUT_CSV_FILE = 'criteria_fs8_input_v2.csv'
OUTPUT_CSV_FILE = 'criteria_fs8_output9.csv'
EXAMPLES_PER_ROW = 5

# --- Claude API Client Initialization ---
# This creates a client to interact with the Anthropic (Claude) API.
# Ensure your API key is set correctly. If the key is invalid, this will fail.
try:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
except Exception as e:
    print(f"Error initializing Anthropic client: {e}")
    print("Please ensure your ANTHROPIC_API_KEY is set correctly as an environment variable or in the script.")
    exit()

def read_prompt_template(filepath):
    """
    Reads the prompt template from a specified text file.
    Handles potential file not found errors.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt template file not found at '{filepath}'")
        return None

def generate_content_with_claude(prompt):
    """
    Sends a formatted prompt to the Claude API and returns the generated transcript and JSON summary.
    Includes error handling for the API call and response parsing.
    """
    if not prompt:
        return None, None

    try:
        # This sends the request to the specified Claude model.
        message = client.messages.create(
            model="claude-sonnet-4-20250514", # Using a strong, generally available model
            max_tokens=8192,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        full_response = message.content[0].text
        
        transcript = None
        json_summary = None

        # --- Corrected Response Parsing for Transcript -> JSON order ---
        if "JSON Summary:" in full_response:
            # Split the response into the Transcript part and the JSON part
            parts = full_response.split("JSON Summary:", 1)
            transcript_part = parts[0]
            json_part = parts[1]

            # Clean up the transcript part
            if transcript_part.strip().startswith("Transcript:"):
                transcript = transcript_part.strip()[len("Transcript:"):].strip()
            else:
                transcript = transcript_part.strip()

            # Clean up and load the JSON part
            json_cleaned = json_part.strip().replace('```json', '').replace('```', '').strip()
            try:
                # Load and dump to ensure it's valid JSON and get a clean string representation.
                json_data = json.loads(json_cleaned)
                json_summary = json.dumps(json_data, indent=2)
            except json.JSONDecodeError:
                print(f"  - Warning: Could not parse JSON summary. Storing as raw text.")
                json_summary = json_cleaned # Store the raw text if parsing fails
        else:
            # Fallback if the "JSON Summary:" separator isn't found
            print("  - Warning: 'JSON Summary:' separator not found. The entire response will be treated as the transcript.")
            transcript = full_response.strip()
            json_summary = "" # Set JSON to empty string if not found

        return transcript, json_summary

    except anthropic.APIError as e:
        print(f"An Anthropic API error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during API call: {e}")
    return None, None

def process_csv(input_filepath, output_filepath, prompt_template):
    """
    Main function to read the input CSV, generate content, and write to the output CSV.
    """
    try:
        # Open the output file immediately to start writing
        with open(output_filepath, 'w', newline='', encoding='utf-8') as outfile:
            # First, read the headers from the input file to set up the writer
            with open(input_filepath, 'r', newline='', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                try:
                    fieldnames = next(reader)
                except StopIteration:
                    print("Error: Input CSV is empty or has no headers.")
                    return
            
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            # Now, process the input file row by row
            with open(input_filepath, 'r', newline='', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                for i, row in enumerate(reader):
                    print(f"\nProcessing row {i+1}: Goal = {row.get('Goal', 'N/A')}")
                    
                    for j in range(EXAMPLES_PER_ROW):
                        print(f"  - Generating example {j+1}/{EXAMPLES_PER_ROW}...")
                        
                        # --- Prompt Formatting ---
                        formatted_prompt = prompt_template
                        formatted_prompt = formatted_prompt.replace('{category}', row.get('Category', ''))
                        formatted_prompt = formatted_prompt.replace('{goal}', row.get('Goal', ''))

                        generated_transcript, generated_json = generate_content_with_claude(formatted_prompt)

                        if generated_transcript or generated_json:
                            new_row = row.copy()
                            new_row['Transcript (oncology)'] = generated_transcript if generated_transcript else ""
                            if 'JSON' in new_row:
                                new_row['JSON'] = generated_json if generated_json else ""
                            
                            writer.writerow(new_row)
                            # Force write to disk for real-time visibility
                            outfile.flush()
                            print(f"  - Successfully generated and wrote example {j+1}.")
                        else:
                            print(f"  - Failed to generate content for example {j+1}. Skipping.")
            
            print(f"\nProcessing complete. Output written to '{output_filepath}'")

    except FileNotFoundError:
        print(f"Error: Input CSV file not found at '{input_filepath}'")
    except Exception as e:
        print(f"An unexpected error occurred during CSV processing: {e}")

def main():
    """
    Main execution function.
    """
    print("Script started.")
    prompt_template = read_prompt_template(PROMPT_TEMPLATE_FILE)
    
    if prompt_template and ANTHROPIC_API_KEY != "YOUR_ANTHROPIC_API_KEY":
        process_csv(INPUT_CSV_FILE, OUTPUT_CSV_FILE, prompt_template)
    elif not prompt_template:
        print("Script aborted because the prompt template could not be read.")
    else:
        print("Script aborted. Please set your ANTHROPIC_API_KEY.")

if __name__ == "__main__":
    main()
