"""
Stratified Sampling for Consumer Complaints

This script creates a stratified sample of 10,000-15,000 complaints from the filtered dataset,
ensuring proportional representation across all product categories.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def create_stratified_sample(
    input_file: str,
    output_file: str,
    sample_size: int = 12000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create a stratified sample from the filtered complaints dataset.
    
    Args:
        input_file: Path to the filtered complaints CSV file
        output_file: Path to save the stratified sample
        sample_size: Target sample size (default: 12000)
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame containing the stratified sample
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"\nProduct distribution in original dataset:")
    print(df['Product'].value_counts())
    print(f"\nProduct distribution (%):")
    print(df['Product'].value_counts(normalize=True) * 100)
    
    # Calculate samples per product category (proportional stratification)
    product_counts = df['Product'].value_counts()
    total_records = len(df)
    
    samples_per_product = {}
    for product, count in product_counts.items():
        proportion = count / total_records
        samples_per_product[product] = int(sample_size * proportion)
    
    # Adjust to ensure we get exactly sample_size records
    total_samples = sum(samples_per_product.values())
    if total_samples < sample_size:
        # Add remaining samples to the largest category
        largest_product = product_counts.index[0]
        samples_per_product[largest_product] += (sample_size - total_samples)
    
    print(f"\nTarget samples per product:")
    for product, count in samples_per_product.items():
        print(f"  {product}: {count}")
    
    # Create stratified sample
    sampled_dfs = []
    for product, n_samples in samples_per_product.items():
        product_df = df[df['Product'] == product]
        if len(product_df) >= n_samples:
            sampled = product_df.sample(n=n_samples, random_state=random_state)
        else:
            # If we don't have enough samples, take all available
            sampled = product_df
            print(f"Warning: Only {len(product_df)} samples available for {product}, requested {n_samples}")
        sampled_dfs.append(sampled)
    
    # Combine all samples
    sample_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # Shuffle the combined sample
    sample_df = sample_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\nSample dataset shape: {sample_df.shape}")
    print(f"\nProduct distribution in sample:")
    print(sample_df['Product'].value_counts())
    print(f"\nProduct distribution in sample (%):")
    print(sample_df['Product'].value_counts(normalize=True) * 100)
    
    # Save the sample
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output_file, index=False)
    print(f"\nStratified sample saved to {output_file}")
    
    return sample_df


def main():
    """Main function to create stratified sample."""
    # Define paths
    input_file = "data/processed/filtered_complaints.csv"
    output_file = "data/processed/sample_complaints.csv"
    
    # Create stratified sample
    sample_df = create_stratified_sample(
        input_file=input_file,
        output_file=output_file,
        sample_size=12000,  # Middle of the 10k-15k range
        random_state=42
    )
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SAMPLING SUMMARY")
    print("="*80)
    print(f"Total samples: {len(sample_df)}")
    print(f"Products represented: {sample_df['Product'].nunique()}")
    print(f"Companies represented: {sample_df['Company'].nunique()}")
    print(f"States represented: {sample_df['State'].nunique()}")
    print(f"Date range: {sample_df['Date received'].min()} to {sample_df['Date received'].max()}")


if __name__ == "__main__":
    main()