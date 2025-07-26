import csv
import os
import sys
import requests
import json
import time
import argparse
from datetime import datetime

def call_claude_api(prompt_content, api_key, max_retries=3, retry_delay=5):
    """Send a prompt to Claude 3.7 Sonnet API and return the response"""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    data = {
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": 4000,
        "messages": [
            {"role": "user", "content": prompt_content}
        ]
    }
    
    retries = 0
    while retries <= max_retries:
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.json()["content"][0]["text"]
            elif response.status_code in [429, 500, 502, 503, 504, 529] and retries < max_retries:
                # Retryable server errors
                wait_time = retry_delay * (2 ** retries)  # Exponential backoff
                print(f"  Received status code {response.status_code}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
                continue
            else:
                print(f"API call failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                return f"Error: API call failed with status code {response.status_code}\nDetails: {response.text}"
        except requests.exceptions.RequestException as e:
            if retries < max_retries:
                wait_time = retry_delay * (2 ** retries)
                print(f"  Request exception: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
                continue
            else:
                print(f"Exception during API call: {e}")
                return f"Error: Network or connection issue: {str(e)}"
        except Exception as e:
            print(f"Unexpected exception during API call: {e}")
            return f"Error: {str(e)}"
    
    return "Error: Maximum retries exceeded. Could not successfully call the API."

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate oncology consultation prompts and optionally send to Claude API')
    
    parser.add_argument('--no-api', action='store_true', 
                        help='Only generate prompts, do not send to Claude API')
    
    parser.add_argument('--limit', type=int, default=None, 
                        help='Limit number of rows to process (default: all rows)')
    
    parser.add_argument('--start-row', type=int, default=1,
                        help='Start processing from this row number in the CSV (default: 1, first data row)')
    
    parser.add_argument('--api-key', type=str, default=None,
                        help='Directly provide the Anthropic API key (alternative to environment variable)')
    
    parser.add_argument('--retry-delay', type=int, default=5,
                        help='Initial delay between retries in seconds (default: 5)')
    
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum number of retries for API calls (default: 3)')
    
    parser.add_argument('--csv-path', type=str, default=os.path.join('data', 'dialogue_list2.csv'),
                        help='Path to the CSV file (default: data/dialogue_list2.csv)')
    
    return parser.parse_args()

def find_column_match(header_row, possible_names):
    """Find the first matching column name from a list of possible names"""
    for possible_name in possible_names:
        if possible_name in header_row:
            return possible_name
    return None

def get_column_value(row, possible_names, default=""):
    """Get value from first matching column name or return default"""
    for col_name in possible_names:
        if col_name in row and row[col_name] and row[col_name].strip():
            return row[col_name].strip()
    return default

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Record the start time for the entire process
    start_time = datetime.now()
    
    # Check for API key if we're calling the API
    api_key = None
    if not args.no_api:
        # Try to get API key from command line argument first, then environment variable
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("WARNING: No API key provided. Please either:")
            print("1. Set environment variable: export ANTHROPIC_API_KEY='your_api_key'")
            print("2. Provide via command line: --api-key='your_api_key'")
            print("\nContinuing with prompt generation only...")
            args.no_api = True
    
    # Extract CSV filename (without extension) for subdirectory names
    csv_filename = os.path.basename(args.csv_path)
    csv_name_without_ext = os.path.splitext(csv_filename)[0]
    
    # Define file paths based on your directory structure
    template_path = os.path.join('prompts', 'template.txt')
    csv_path = args.csv_path  # Use the CSV path from command line arguments
    
    # Create subdirectories based on CSV filename
    output_dir = os.path.join('dialogue_prompts', csv_name_without_ext)
    dialogue_results_dir = os.path.join('dialogue_results', csv_name_without_ext)
    
    print(f"Will save prompts to: {output_dir}/")
    print(f"Will save responses to: {dialogue_results_dir}/")
    
    # Read the template file
    try:
        with open(template_path, 'r') as file:
            template_content = file.read()
        print(f"Successfully read the template file from {template_path}")
    except Exception as e:
        print(f"Error reading template file: {e}")
        return
    
    # Read the CSV file
    try:
        with open(csv_path, 'r') as file:
            csv_reader = csv.DictReader(file)
            header_row = csv_reader.fieldnames
            all_csv_rows = list(csv_reader)
            
        total_csv_rows = len(all_csv_rows)
        print(f"CSV file contains {total_csv_rows} data rows")
        
        # Apply start row filter (start_row is 1-indexed for the first data row after header)
        if args.start_row > 1:
            start_index = args.start_row - 1  # Convert to 0-indexed
            if start_index >= total_csv_rows:
                print(f"ERROR: Start row {args.start_row} exceeds the number of rows in the CSV ({total_csv_rows})")
                return
            
            rows_to_skip = start_index
            csv_rows = all_csv_rows[start_index:]
            print(f"Starting from CSV row {args.start_row} (skipping {rows_to_skip} rows)")
        else:
            csv_rows = all_csv_rows
        
        # Apply row limit if specified
        original_row_count = len(csv_rows)
        if args.limit is not None and args.limit > 0:
            csv_rows = csv_rows[:args.limit]
            print(f"Limited to processing {args.limit} rows (out of {original_row_count} remaining rows)")
        
        rows_to_process = len(csv_rows)
        print(f"Will process {rows_to_process} rows from the CSV file at {csv_path}")
        print(f"CSV columns found: {header_row}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Define possible column names for each required field
    interaction_type_cols = ["Interaction Type", "InteractionType", "Visit Type", "VisitType", "Interaction"]
    provider_type_cols = ["Provider Type", "ProviderType", "Provider", "Doctor Type", "DoctorType"]
    visit_length_cols = ["Approx visit length (min)", "Visit Length", "VisitLength", "Length", "Duration", "Visit Duration"]
    cancer_type_cols = ["Cancer Type", "CancerType", "Type of Cancer", "Cancer", "Malignancy"]
    distractor_cols = ["Distraction added to script?", "Distractor Element", "Distractor", "Distraction"]
    script_name_cols = ["Script Name", "ScriptName", "Name", "Filename", "Output Name"]
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Create dialogue_results directory if we're calling the API
    if not args.no_api and not os.path.exists(dialogue_results_dir):
        os.makedirs(dialogue_results_dir)
        print(f"Created directory: {dialogue_results_dir}")
    
    # Define the distractor section to append when a distractor is present
    distractor_section = """
## Distractor Element

The dialogue should incorporate the following distractor element that creates a realistic diversion from the medical discussion:

"{distractor_text}"

This distractor should:
- Be naturally integrated into the conversation
- Create a realistic challenge for the provider to address while staying on topic
- Show appropriate provider response to the distraction
- Eventually allow the consultation to return to necessary medical discussions
"""
    
    # Process each row and generate custom prompts
    processed_count = 0
    success_count = 0
    error_count = 0
    
    for i, row in enumerate(csv_rows):
        row_start_time = datetime.now()
        
        # Calculate the actual row number in the original CSV (including header row)
        csv_row_number = i + args.start_row + 1  # +1 for header row
        
        # Calculate progress percentage
        progress_pct = (i+1) / rows_to_process * 100
        
        # Enhanced progress output
        print(f"\n[{progress_pct:.1f}%] Processing CSV row {csv_row_number}/{total_csv_rows+1} (item {i+1} of {rows_to_process})...")
        
        # Get values using the flexible column mapping
        interaction_type = get_column_value(row, interaction_type_cols, "Initial Consultation")
        provider_type = get_column_value(row, provider_type_cols, "Oncologist")
        visit_length = get_column_value(row, visit_length_cols, "30")
        cancer_type = get_column_value(row, cancer_type_cols, "")
        distractor = get_column_value(row, distractor_cols, "")
        script_name = get_column_value(row, script_name_cols, "")
        
        # Log the extracted values for debugging
        print(f"  Extracted values:")
        print(f"    Interaction Type: {interaction_type}")
        print(f"    Provider Type: {provider_type}")
        print(f"    Visit Length: {visit_length}")
        print(f"    Cancer Type: {cancer_type}")
        print(f"    Distractor: {distractor}")
        
        # Replace all placeholders in the template
        custom_prompt = template_content
        custom_prompt = custom_prompt.replace("[INTERACTION_TYPE]", interaction_type)
        custom_prompt = custom_prompt.replace("[PROVIDER_TYPE]", provider_type)
        custom_prompt = custom_prompt.replace("[CANCER_TYPE]", cancer_type)
        custom_prompt = custom_prompt.replace("[VISIT_LENGTH]", visit_length)
        
        # Append the distractor section only if a distractor is present
        if distractor:
            # Format the distractor section with the specific distractor text
            formatted_distractor_section = distractor_section.format(distractor_text=distractor)
            
            # Append it to the custom prompt
            custom_prompt += formatted_distractor_section
            
            print(f"  Added distractor: '{distractor}'")
        
        # Generate script name if not provided
        if not script_name:
            # Create a generic name if Script Name is empty
            script_name = f"{interaction_type}_{provider_type}_{cancer_type}".replace(" ", "_")
        
        # Always prepend the row number to the filename - use the actual CSV row number
        prefixed_script_name = f"Row{csv_row_number}_{script_name}"
        
        # Clean the filename to remove any problematic characters
        filename = f"{prefixed_script_name}.txt"
        filename = filename.replace("/", "_").replace("\\", "_").replace(":", "_")
        prompt_file_path = os.path.join(output_dir, filename)
        
        with open(prompt_file_path, 'w') as file:
            file.write(custom_prompt)
        
        print(f"  Saved prompt to: {prompt_file_path}")
        processed_count += 1
        
        # Send to Claude API if flag is not set
        if not args.no_api:
            print(f"  Sending prompt to Claude 3.7 Sonnet API...")
            api_start_time = datetime.now()
            
            response = call_claude_api(
                prompt_content=custom_prompt, 
                api_key=api_key,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay
            )
            
            api_duration = (datetime.now() - api_start_time).total_seconds()
            
            # Save Claude's response - also with row number prefix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            response_filename = f"Row{csv_row_number}_{script_name}_response_{timestamp}.txt"
            response_filename = response_filename.replace("/", "_").replace("\\", "_").replace(":", "_")
            response_file_path = os.path.join(dialogue_results_dir, response_filename)
            
            with open(response_file_path, 'w') as file:
                file.write(response)
            
            # Check if the response was an error
            if response.startswith("Error:"):
                print(f"  WARNING: Received error response. Check {response_file_path} for details.")
                print(f"  API call took {api_duration:.1f} seconds but failed.")
                error_count += 1
            else:
                print(f"  Successfully saved Claude's response to: {response_file_path}")
                print(f"  API call completed in {api_duration:.1f} seconds.")
                success_count += 1
            
            # Add a larger delay to avoid hitting rate limits
            if i < len(csv_rows) - 1:
                delay = 5  # Increased from 2 seconds to 5 seconds
                print(f"  Waiting {delay} seconds before next request...")
                time.sleep(delay)
        
        # Show timing for this row
        row_duration = (datetime.now() - row_start_time).total_seconds()
        print(f"  Row {csv_row_number} completed in {row_duration:.1f} seconds.")
    
    # Calculate total elapsed time
    total_duration = (datetime.now() - start_time).total_seconds()
    minutes, seconds = divmod(total_duration, 60)
    hours, minutes = divmod(minutes, 60)
    
    # Final summary
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Start row: {args.start_row}")
    if args.limit:
        print(f"Limit: {args.limit} rows")
    print(f"Total rows processed: {processed_count}")
    if not args.no_api:
        print(f"Successful API calls: {success_count}")
        print(f"Failed API calls: {error_count}")
    print(f"Total processing time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    
    print(f"\nPrompts saved to: {output_dir}/")
    if not args.no_api:
        print(f"Responses saved to: {dialogue_results_dir}/")

if __name__ == "__main__":
    main()