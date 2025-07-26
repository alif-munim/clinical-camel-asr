import os
import csv
import re
import glob
import argparse
from datetime import datetime

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Update CSV with dialogue responses from Claude')
    
    parser.add_argument('--csv-path', type=str, default=os.path.join('data', 'dialogue_list3.csv'),
                        help='Path to the original CSV file (default: data/dialogue_list3.csv)')
    
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Path to save the updated CSV (default: adds "_updated" to original filename)')
    
    parser.add_argument('--responses-dir', type=str, default=None,
                        help='Directory containing response files (default: derived from CSV filename)')
    
    parser.add_argument('--column-name', type=str, default="Cloud",
                        help='Name of the column to update (default: "Cloud")')
    
    return parser.parse_args()

def extract_row_number(filename):
    """Extract row number from a filename that starts with RowX_"""
    match = re.match(r'Row(\d+)_', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return None

def get_timestamp_from_filename(filename):
    """Extract timestamp from a response filename"""
    # Pattern to match YYYYMMDD_HHMMSS format
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        timestamp_str = match.group(1)
        try:
            return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except ValueError:
            pass
    return None

def get_most_recent_response_for_row(responses_dir, row_number):
    """Get the most recent response file for a given row number"""
    pattern = os.path.join(responses_dir, f"Row{row_number}_*_response_*.txt")
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        return None
    
    # Sort files by timestamp (most recent last)
    sorted_files = sorted(matching_files, 
                         key=lambda f: get_timestamp_from_filename(f) or datetime.min)
    
    # Return the most recent file
    return sorted_files[-1] if sorted_files else None

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Extract CSV filename (without extension) for directory names
    csv_filename = os.path.basename(args.csv_path)
    csv_name_without_ext = os.path.splitext(csv_filename)[0]
    
    # Determine output CSV path if not specified
    if not args.output_csv:
        output_dir = os.path.dirname(args.csv_path) or '.'
        output_filename = f"{csv_name_without_ext}_updated.csv"
        output_csv_path = os.path.join(output_dir, output_filename)
    else:
        output_csv_path = args.output_csv
    
    # Determine responses directory if not specified
    if not args.responses_dir:
        responses_dir = os.path.join('dialogue_results', csv_name_without_ext)
    else:
        responses_dir = args.responses_dir
    
    print(f"Looking for response files in: {responses_dir}")
    print(f"Will update column: {args.column_name}")
    print(f"Will save updated CSV to: {output_csv_path}")
    
    # Check if responses directory exists
    if not os.path.exists(responses_dir):
        print(f"ERROR: Responses directory not found: {responses_dir}")
        return
    
    # Check if original CSV exists
    if not os.path.exists(args.csv_path):
        print(f"ERROR: Original CSV file not found: {args.csv_path}")
        return
    
    # Get all response files in the directory
    all_response_files = glob.glob(os.path.join(responses_dir, "Row*_*_response_*.txt"))
    print(f"Found {len(all_response_files)} response files")
    
    # Extract unique row numbers from filenames
    row_numbers = set()
    for filename in all_response_files:
        row_num = extract_row_number(filename)
        if row_num:
            row_numbers.add(row_num)
    
    print(f"Found responses for {len(row_numbers)} unique rows")
    
    # Read the original CSV
    csv_rows = []
    with open(args.csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        
        # Check if the column exists
        if args.column_name not in fieldnames:
            print(f"WARNING: Column '{args.column_name}' not found in CSV. It will be added.")
            fieldnames.append(args.column_name)
        
        # Read all rows
        for row in reader:
            csv_rows.append(row)
    
    print(f"Read {len(csv_rows)} rows from original CSV")
    
    # Track statistics
    updated_count = 0
    skipped_count = 0
    
    # Update the rows with response content
    for i, row in enumerate(csv_rows):
        # CSV rows are 0-indexed in the list, but 1-indexed in the file (plus header)
        row_number = i + 2  # +1 for header, +1 for 1-indexing
        
        # Get the most recent response file for this row
        response_file = get_most_recent_response_for_row(responses_dir, row_number)
        
        if response_file:
            try:
                with open(response_file, 'r', encoding='utf-8') as f:
                    response_content = f.read()
                
                # Update the row with the response content
                row[args.column_name] = response_content
                updated_count += 1
                
                print(f"Row {row_number}: Updated with content from {os.path.basename(response_file)}")
            except Exception as e:
                print(f"Error reading response file for row {row_number}: {e}")
                skipped_count += 1
        else:
            print(f"Row {row_number}: No response file found")
            skipped_count += 1
    
    # Write the updated CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print("\n=== UPDATE COMPLETE ===")
    print(f"Rows updated: {updated_count}")
    print(f"Rows skipped: {skipped_count}")
    print(f"Updated CSV saved to: {output_csv_path}")

if __name__ == "__main__":
    main()