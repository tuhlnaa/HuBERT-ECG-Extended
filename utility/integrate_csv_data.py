"""
Script to integrate label data from multiple CSV files in a deep learning dataset.
Removes duplicate entries based on filename and consolidates all data into a single CSV.

Usage:
python utility/integrate_csv_data.py /path/to/dataset --output integrated_dataset.csv
"""

import argparse
import pandas as pd

from pathlib import Path
from typing import Optional


def save_summary_to_file(df, output_path: Path, output_file: str) -> None:
    """Save dataset summary statistics to a text file."""
    # Prepare summary statistics to save to file
    summary_lines = []
    summary_lines.append("=== Dataset Summary ===")
    summary_lines.append(f"Total unique samples: {len(df)}")
    summary_lines.append(f"Columns: {list(df.columns)}")
    
    # Count non-zero tags for each label column (excluding filename, age, sex)
    tag_columns = [col for col in df.columns if col not in ['filename', 'age', 'sex']]
    if tag_columns:
        summary_lines.append("")
        summary_lines.append("=== Tag Distribution ===")
        for col in tag_columns:
            if df[col].dtype in ['int64', 'float64']:
                count = (df[col] == 1).sum()
                percentage = (count / len(df)) * 100
                summary_lines.append(f"{col}: {count} ({percentage:.2f}%)")
    
    # Save summary to file
    summary_file = output_path / output_file.replace('.csv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"Summary statistics saved to: {summary_file}")


def integrate_csv_files(
    dataset_path: Path | str,
    output_file: str = 'integrated_dataset.csv',
    pattern: str = '**/*.csv'
) -> Optional[pd.DataFrame]:
    """
    Integrate multiple CSV files into a single CSV, removing duplicates.
    
    Args:
        dataset_path: Path to the dataset directory
        output_file: Name of the output CSV file
        pattern: Glob pattern to match CSV files (default: '**/*.csv' for recursive search)
    
    Returns:
        The integrated dataframe, or None if no CSV files found
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
    print(f"Total rows after deduplication: {len(deduplicated_df)}")
    
    # Sort by filename for consistency
    deduplicated_df = deduplicated_df.sort_values(by=filename_column).reset_index(drop=True)
    
    # Save to output file
    output_path = dataset_path / output_file
    deduplicated_df.to_csv(output_path, index=False)
    print(f"\nIntegrated dataset saved to: {output_path}")
    
    # Save summary statistics to a separate text file
    save_summary_to_file(deduplicated_df, dataset_path, output_file)
    
    return deduplicated_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Integrate CSV tag data from multiple files, removing duplicates.'
    )
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