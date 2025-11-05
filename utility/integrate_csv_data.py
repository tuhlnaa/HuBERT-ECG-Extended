"""
Script to integrate label data from multiple CSV files in a deep learning dataset.
Removes duplicate entries based on filename and consolidates all data into a single CSV.

Uusage:
python utility/integrate_csv_data.py /path/to/dataset --output integrated_dataset.csv
"""

import argparse
import pandas as pd
from pathlib import Path


def integrate_csv_files(dataset_path, output_file='integrated_dataset.csv', pattern='**/*.csv'):
    """
    Integrate multiple CSV files into a single CSV, removing duplicates.
    
    Args:
        dataset_path (str or Path): Path to the dataset directory
        output_file (str): Name of the output CSV file
        pattern (str): Glob pattern to match CSV files (default: '**/*.csv' for recursive search)
    
    Returns:
        pd.DataFrame: The integrated dataframe
    """
    dataset_path = Path(dataset_path)
    
    # Find all CSV files matching the pattern
    csv_files = list(dataset_path.glob(pattern))
    
    if not csv_files:
        print(f"No CSV files found in {dataset_path} matching pattern {pattern}")
        return None
    
    print(f"Found {len(csv_files)} CSV files.")
    
    # Read and concatenate all CSV files
    dataframes = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"\nTotal rows before deduplication: {len(combined_df)}")
    
    # Remove duplicates based on filename (assuming 'filename' is the first column)
    # Keep the first occurrence
    filename_column = combined_df.columns[0]
    deduplicated_df = combined_df.drop_duplicates(subset=[filename_column], keep='first')
    
    duplicates_removed = len(combined_df) - len(deduplicated_df)
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Total rows after deduplication: {len(deduplicated_df)}")
    
    # Sort by filename for consistency
    deduplicated_df = deduplicated_df.sort_values(by=filename_column).reset_index(drop=True)
    
    # Save to output file
    output_path = dataset_path / output_file
    deduplicated_df.to_csv(output_path, index=False)
    print(f"\nIntegrated dataset saved to: {output_path}")
    
    # Display summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Total unique samples: {len(deduplicated_df)}")
    print(f"Columns: {list(deduplicated_df.columns)}")
    
    # Count non-zero tags for each label column (excluding filename, age, sex)
    tag_columns = [col for col in deduplicated_df.columns if col not in ['filename', 'age', 'sex']]
    if tag_columns:
        print("\n=== Tag Distribution ===")
        for col in tag_columns:
            if deduplicated_df[col].dtype in ['int64', 'float64']:
                count = (deduplicated_df[col] == 1).sum()
                percentage = (count / len(deduplicated_df)) * 100
                print(f"{col}: {count} ({percentage:.2f}%)")
    
    return deduplicated_df


def main():
    parser = argparse.ArgumentParser(description='Integrate CSV tag data from multiple files, removing duplicates.')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset directory (e.g., dataset/ptb/)')
    parser.add_argument('--output', type=str, default='integrated_dataset.csv', help='Output CSV filename (default: integrated_dataset.csv)')
    parser.add_argument('--pattern', type=str, default='*.csv', help='Pattern to match CSV files (default: *.csv). Use **/*.csv for recursive search.')
    
    args = parser.parse_args()
    
    # Convert to Path object and check if it exists
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path '{dataset_path}' does not exist")
        return
    
    # Run integration
    integrate_csv_files(dataset_path, args.output, args.pattern)

if __name__ == '__main__':
    main()