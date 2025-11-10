"""
Script to visualize ECG disease distribution from a CSV dataset.
Creates a bar chart showing disease counts with alphabetical labels and a mapping file.

Usage:
    python utility/ecg_disease_chart.py <csv_file> --output ecg_diseases_chart.png
    python utility/ecg_disease_chart.py <csv_file> --metadata-cols filename age sex
"""

import argparse
import mplcyberpunk
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path
from typing import Dict, List


def generate_labels(n: int) -> List[str]:
    """Generate labels A-Z, then AA-AZ, BA-BZ, etc."""
    labels = []
    for i in range(n):
        if i < 26:
            # A-Z
            labels.append(chr(65 + i))
        else:
            # AA-AZ, BA-BZ, etc.
            first_letter = chr(65 + (i - 26) // 26)
            second_letter = chr(65 + (i - 26) % 26)
            labels.append(first_letter + second_letter)
    return labels


def save_disease_mapping(
    disease_to_label: Dict[str, str],
    disease_counts: pd.Series,
    output_path: Path
) -> None:
    """Save disease label mapping to a text file."""
    mapping_file = output_path.parent / output_path.name.replace('.png', '_mapping.txt')
    
    with open(mapping_file, 'w') as f:
        f.write("=== Disease Label Mapping ===\n")
        f.write("-" * 50 + "\n")
        for disease, label in disease_to_label.items():
            count = disease_counts[disease]
            f.write(f"{label}: {disease} (Count: {int(count)})\n")
    
    print(f"Disease mapping saved to: {mapping_file}")


def plot_ecg_diseases(
    csv_file: Path | str, 
    output_file: str = 'ecg_diseases_chart.png',
    metadata_cols: List[str] | None = None
) -> None:
    """
    Create a bar chart showing the count of ECG diseases.
    
    Args:
        csv_file: Path to the CSV file containing ECG data
        output_file: Name of the output chart file
        metadata_cols: List of column names to exclude from disease columns.
                      Defaults to ['filename', 'age', 'sex']
    """
    csv_file = Path(csv_file)
    output_path = csv_file.parent / output_file

    df = pd.read_csv(csv_file)
    
    # Use default metadata columns if not provided
    if metadata_cols is None:
        metadata_cols = ['filename', 'age', 'sex', "Patient_ID", "strat_fold"]
    
    # Get disease columns (exclude metadata columns)
    disease_cols = [col for col in df.columns if col not in metadata_cols]
    
    if not disease_cols:
        print("Error: No disease columns found in the CSV file")
        print(f"Available columns: {list(df.columns)}")
        print(f"Excluded metadata columns: {metadata_cols}")
        return
    
    print(f"Found {len(disease_cols)} disease columns")
    print(f"Excluded metadata columns: {metadata_cols}")
    
    # Count occurrences of each disease
    disease_counts = df[disease_cols].sum().sort_values(ascending=False)

    # Generate alphabetical labels
    labels = generate_labels(len(disease_counts))
    
    # Create a mapping from disease names to labels
    disease_to_label = {disease: label for disease, label in zip(disease_counts.index, labels)}
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Disease': labels,
        'Count': disease_counts.values
    })

    # Set up the plot style
    plt.style.use("cyberpunk")
    if len(disease_counts) > 15:
        plt.figure(figsize=(14, 6))
    else:
        plt.figure()

    # Define color palette (Data volume ranking)
    pal = sns.color_palette('coolwarm_r', len(disease_counts))

    # Create bar chart
    ax = sns.barplot(
        data=plot_df, 
        x='Disease', 
        y='Count', 
        palette=pal,
        hue='Disease', width=0.6
    )
    
    plt.title('Number of ECG Diseases', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Disease Label', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    
    if len(disease_counts) > 15:
        plt.xticks(rotation=45, ha='right')
    else:
        # Add value labels on top of bars
        for i, (count, label) in enumerate(zip(disease_counts.values, labels)):
            ax.text(i, count, str(int(count)), 
                    ha='center', va='bottom', fontsize=14)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Chart saved to: {output_path}")
    
    # Save the disease mapping to file
    save_disease_mapping(disease_to_label, disease_counts, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Create a bar chart visualization of ECG disease distribution.'
    )
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing ECG data')
    parser.add_argument('--output', type=str, default='ecg_diseases_chart.png', help='Output chart filename (default: ecg_diseases_chart.png)')
    parser.add_argument('--metadata-cols', nargs='+', default=None, help='Column names to exclude from disease columns (default: filename age sex)')
    args = parser.parse_args()
    
    # Convert to Path object and check if it exists
    csv_file = Path(args.csv_file)
    if not csv_file.exists():
        print(f"Error: CSV file '{csv_file}' does not exist")
        return
    
    plot_ecg_diseases(csv_file, args.output, args.metadata_cols)


if __name__ == '__main__':
    main()