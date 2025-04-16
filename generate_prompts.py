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
    
    return parser.parse_args()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate oncology consultation prompts and optionally send to Claude API')
    
    parser.add_argument('--no-api', action='store_true', 
                        help='Only generate prompts, do not send to Claude API')
    
    parser.add_argument('--limit', type=int, default=None, 
                        help='Limit number of rows to process (default: all rows)')
    
    parser.add_argument('--api-key', type=str, default=None,
                        help='Directly provide the Anthropic API key (alternative to environment variable)')
    
    parser.add_argument('--retry-delay', type=int, default=5,
                        help='Initial delay between retries in seconds (default: 5)')
    
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum number of retries for API calls (default: 3)')
    
    parser.add_argument('--csv-path', type=str, default=os.path.join('data', 'dialogue_list2.csv'),
                        help='Path to the CSV file (default: data/dialogue_list2.csv)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
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
    
    # Define file paths based on your directory structure
    template_path = os.path.join('prompts', 'template.txt')
    csv_path = args.csv_path  # Use the CSV path from command line arguments
    output_dir = os.path.join('prompts')
    dialogue_results_dir = os.path.join('dialogue_results')
    
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
            csv_rows = list(csv_reader)
        
        # Apply row limit if specified
        if args.limit is not None and args.limit > 0:
            csv_rows = csv_rows[:args.limit]
            print(f"Limited to processing first {args.limit} rows of the CSV file")
        
        print(f"Will process {len(csv_rows)} rows from the CSV file at {csv_path}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Create dialogue_results directory if we're calling the API
    if not args.no_api and not os.path.exists(dialogue_results_dir):
        os.makedirs(dialogue_results_dir)
        print(f"Created directory: {dialogue_results_dir}")
    
    # Process each row and generate custom prompts
    for i, row in enumerate(csv_rows):
        print(f"\nProcessing row {i+1}/{len(csv_rows)}...")
        
        # Replace all placeholders in the template
        custom_prompt = template_content
        custom_prompt = custom_prompt.replace("[INTERACTION_TYPE]", row["Interaction Type"])
        custom_prompt = custom_prompt.replace("[PROVIDER_TYPE]", row["Provider Type"])
        custom_prompt = custom_prompt.replace("[CANCER_TYPE]", row["Cancer Type"])
        custom_prompt = custom_prompt.replace("[VISIT_LENGTH]", row["Approx visit length (min)"])
        
        # Add any distraction information if applicable
        if "Distraction added to script?" in row and row["Distraction added to script?"] and row["Distraction added to script?"].strip():
            distraction_note = f"\n\nNote: Include discussion of {row['Distraction added to script?']} as a brief distraction in the dialogue."
            custom_prompt += distraction_note
        
        # Save to file
        script_name = row["Script Name"].strip() if "Script Name" in row else ""
        if not script_name:
            # Create a generic name if Script Name is empty
            script_name = f"Script_{row['Interaction Type']}_{row['Provider Type']}_{row['Cancer Type']}".replace(" ", "_")
        
        # Clean the filename to remove any problematic characters
        filename = f"{script_name}.txt"
        filename = filename.replace("/", "_").replace("\\", "_").replace(":", "_")
        prompt_file_path = os.path.join(output_dir, filename)
        
        with open(prompt_file_path, 'w') as file:
            file.write(custom_prompt)
        
        print(f"  Saved prompt to: {prompt_file_path}")
        
        # Send to Claude API if flag is not set
        if not args.no_api:
            print(f"  Sending prompt to Claude 3.7 Sonnet API...")
            response = call_claude_api(
                prompt_content=custom_prompt, 
                api_key=api_key,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay
            )
            
            # Save Claude's response
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            response_filename = f"{script_name}_response_{timestamp}.txt"
            response_filename = response_filename.replace("/", "_").replace("\\", "_").replace(":", "_")
            response_file_path = os.path.join(dialogue_results_dir, response_filename)
            
            with open(response_file_path, 'w') as file:
                file.write(response)
            
            # Check if the response was an error
            if response.startswith("Error:"):
                print(f"  WARNING: Received error response. Check {response_file_path} for details.")
            else:
                print(f"  Successfully saved Claude's response to: {response_file_path}")
            
            # Add a larger delay to avoid hitting rate limits
            if i < len(csv_rows) - 1:
                delay = 5  # Increased from 2 seconds to 5 seconds
                print(f"  Waiting {delay} seconds before next request...")
                time.sleep(delay)
    
    if args.no_api:
        print("\nAll prompts have been generated in the prompts/ directory!")
    else:
        print("\nAll prompts have been processed and dialogues saved to the dialogue_results/ directory!")

if __name__ == "__main__":
    main()