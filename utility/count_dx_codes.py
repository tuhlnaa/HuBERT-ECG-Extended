"""
Script to count diagnosis codes from HEA files and update CSV file with counts.

This script scans a dataset directory for HEA files, extracts diagnosis codes from
the #Dx: lines, counts their occurrences, and updates a CSV file with the counts.

The CSV file should contain at least a 'dx_code' column. The script will add or
update a 'count' column with the number of times each diagnosis code appears in
the dataset.

Usage:
    python utility/count_dx_codes.py --root /path/to/dataset --csv diagnoses.csv
    python utility/count_dx_codes.py --root /path/to/dataset --csv diagnoses.csv --output updated.csv
"""

import argparse
import csv
import wfdb
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set


def parse_hea_file(hea_path: Path) -> Set[str]:
    """Parse a HEA file and extract diagnosis codes using wfdb."""
    dx_codes = set()
    
    try:
        # Read the record header (without the .hea extension)
        record_name = str(hea_path.with_suffix(''))
        record = wfdb.rdheader(record_name)
        
        # Extract diagnosis codes from comments
        for comment in record.comments:
            # Handle both '#Dx:' format and 'Dx:' format
            comment_stripped = comment.lstrip('#').strip()
            if comment_stripped.startswith('Dx:'):
                codes_str = comment_stripped.split(':', 1)[1].strip()
                if codes_str:
                    codes = [code.strip() for code in codes_str.split(',')]
                    dx_codes.update(codes)
                break
    except Exception as e:
        print(f"Warning: Error reading {hea_path}: {e}")
    
    return dx_codes


def scan_dataset(root_dir: Path) -> Counter:
    """Scan all HEA files in the root directory and count diagnosis codes."""
    dx_counter = Counter()
    hea_files = list(root_dir.rglob('*.hea'))
    
    if not hea_files:
        print(f"Warning: No HEA files found in {root_dir}")
        return dx_counter
    
    print(f"Found {len(hea_files)} HEA files")
    
    for hea_file in hea_files:
        dx_codes = parse_hea_file(hea_file)
        dx_counter.update(dx_codes)
    
    return dx_counter


def update_csv(csv_path: Path, dx_counts: Counter, output_path: Path = None) -> None:
    """Update CSV file with diagnosis code counts."""
    if output_path is None:
        output_path = csv_path
    
    # Read the CSV file
    rows = []
    fieldnames = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            if not fieldnames:
                print(f"Error: CSV file {csv_path} appears to be empty")
                return
            
            # Add 'count' field if not present
            if 'count' not in fieldnames:
                fieldnames = list(fieldnames) + ['count']
            
            for row in reader:
                dx_code = row.get('dx_code', '').strip()
                # Add count for this diagnosis code
                row['count'] = dx_counts.get(dx_code, 0)
                rows.append(row)
    
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Write updated CSV
    try:
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\nSuccessfully updated CSV file: {output_path}")
        print(f"Total diagnosis codes processed: {len(rows)}")
        print(f"Total diagnosis occurrences counted: {sum(dx_counts.values())}")
        
    except Exception as e:
        print(f"Error writing CSV file: {e}")


def main():
    """Main function to parse arguments and orchestrate the process."""
    parser = argparse.ArgumentParser(
        description='Count diagnosis codes from HEA files and update CSV file',
    )
    parser.add_argument('--root', type=Path, required=True, help='Root directory containing HEA files')
    parser.add_argument('--csv', type=Path, required=True, help='Path to CSV file with diagnosis codes')
    parser.add_argument('--output', type=Path, default=None, help='Path to output CSV file (default: update input CSV file)')
    args = parser.parse_args()
    
    # Process
    print(f"Scanning dataset in: {args.root}")
    dx_counts = scan_dataset(args.root)
    print(f"\nFound {len(dx_counts)} unique diagnosis codes")
    
    # Show top 10 most common codes
    if dx_counts:
        print("\nTop 10 most common diagnosis codes:")
        for code, count in dx_counts.most_common(10):
            print(f"  {code}: {count}")
    
    # Update CSV
    print(f"\nUpdating CSV file: {args.csv}")
    update_csv(args.csv, dx_counts, args.output)
    
    return 0


if __name__ == '__main__':
    main()