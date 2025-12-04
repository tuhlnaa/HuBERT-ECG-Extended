"""
Script to analyze Boolean and numeric columns from CSV files.

This script reads a CSV file and analyzes selected columns, automatically detecting
whether each column contains Boolean or numeric values. For Boolean columns, it
calculates true/false counts and percentages. For numeric columns, it calculates
maximum, minimum, and average values.

The script provides a clean, formatted output to the terminal showing statistics
for each selected column.

Usage:
    python utility/csv_analyzer.py <csv_file> --columns <column1> <column2> ...
    python utility/csv_analyzer.py data.csv --columns normal_ecg death timey
"""

import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any


def is_boolean_column(values: List[str]) -> bool:
    """Check if column contains boolean values."""
    unique_values = set(v.lower() for v in values if v.strip())
    return unique_values.issubset({'true', 'false', '1', '0', 'yes', 'no'})


def is_numeric_column(values: List[str]) -> bool:
    """Check if column contains numeric values."""
    try:
        for value in values:
            if value.strip():
                float(value)
        return True
    except ValueError:
        return False


def parse_boolean(value: str) -> bool:
    """Convert string to boolean."""
    return value.lower() in {'true', '1', 'yes'}


def analyze_boolean_column(values: List[str]) -> Dict[str, Any]:
    """Analyze a boolean column and return statistics."""
    bool_values = [parse_boolean(v) for v in values if v.strip()]
    true_count = sum(bool_values)
    false_count = len(bool_values) - true_count
    
    return {
        'type': 'boolean',
        'total': len(bool_values),
        'true_count': true_count,
        'false_count': false_count,
        'true_percentage': (true_count / len(bool_values) * 100) if bool_values else 0
    }


def analyze_numeric_column(values: List[str]) -> Dict[str, Any]:
    """Analyze a numeric column and return statistics."""
    numeric_values = [float(v) for v in values if v.strip()]
    
    if not numeric_values:
        return {'type': 'numeric', 'error': 'No valid numeric values'}
    
    return {
        'type': 'numeric',
        'count': len(numeric_values),
        'maximum': max(numeric_values),
        'minimum': min(numeric_values),
        'average': sum(numeric_values) / len(numeric_values)
    }


def analyze_csv(file_path: Path, selected_columns: List[str]) -> Dict[str, Dict[str, Any]]:
    """Analyze selected columns from a CSV file."""
    results = {}
    
    # Read CSV file
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Validate that selected columns exist
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("CSV file has no header row")
        
        invalid_columns = [col for col in selected_columns if col not in fieldnames]
        if invalid_columns:
            raise ValueError(f"Columns not found in CSV: {', '.join(invalid_columns)}")
        
        # Collect all values for selected columns
        column_data = {col: [] for col in selected_columns}
        for row in reader:
            for col in selected_columns:
                column_data[col].append(row[col])
    
    # Analyze each column
    for column_name, values in column_data.items():
        if is_boolean_column(values):
            results[column_name] = analyze_boolean_column(values)
        elif is_numeric_column(values):
            results[column_name] = analyze_numeric_column(values)
        else:
            results[column_name] = {
                'type': 'unknown',
                'error': 'Column is neither boolean nor numeric'
            }
    
    return results


def print_results(results: Dict[str, Dict[str, Any]]):
    """Print analysis results to terminal."""
    print("\n" + "=" * 60)
    print("CSV ANALYSIS RESULTS")
    print("=" * 60)
    
    for column_name, stats in results.items():
        print(f"\n📊 Column: {column_name}")
        print("-" * 60)
        
        if 'error' in stats:
            print(f"❌ Error: {stats['error']}")
            continue
        
        if stats['type'] == 'boolean':
            print(f"Type: Boolean")
            print(f"Total values: {stats['total']}")
            print(f"True count: {stats['true_count']}")
            print(f"False count: {stats['false_count']}")
            print(f"True percentage: {stats['true_percentage']:.2f}%")
        
        elif stats['type'] == 'numeric':
            print(f"Type: Numeric")
            print(f"Count: {stats['count']}")
            print(f"Maximum: {stats['maximum']:.6f}")
            print(f"Minimum: {stats['minimum']:.6f}")
            print(f"Average: {stats['average']:.6f}")
    
    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze Boolean or numeric columns from a CSV file.',)
    parser.add_argument('csv_file', type=str, help='Path to the CSV file to analyze')
    parser.add_argument('--columns', nargs='+', required=True, help='Column names to analyze (space-separated)')
    args = parser.parse_args()
    
    # Convert to Path object
    csv_path = Path(args.csv_file)
    
    # Analyze CSV
    results = analyze_csv(csv_path, args.columns)
    print_results(results)


if __name__ == '__main__':
    main()