
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PROCESSED_DATA_PATH = 'data/processed/filtered_complaints.csv'
REPORT_STATS_DIR = 'report/stats'
REPORT_IMAGES_DIR = 'report/images'

def generate_eda():
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"File not found: {PROCESSED_DATA_PATH}")
        return

    print("Loading processed data for EDA...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Data shape: {df.shape}")

    # Ensure output directories exist
    os.makedirs(REPORT_STATS_DIR, exist_ok=True)
    os.makedirs(REPORT_IMAGES_DIR, exist_ok=True)

    # 1. Distribution of complaints across products
    print("Generating product distribution plot...")
    plt.figure(figsize=(10, 6))
    sns.countplot(y='Product', data=df, order=df['Product'].value_counts().index)
    plt.title('Distribution of Complaints by Product')
    plt.xlabel('Number of Complaints')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'product_distribution.png'))
    plt.close()

    # 2. Length of narratives
    print("Calculating narrative lengths...")
    df['narrative_length'] = df['cleaned_narrative'].astype(str).apply(lambda x: len(x.split()))
    
    print("Generating narrative length distribution plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['narrative_length'], bins=50, kde=True)
    plt.title('Distribution of Complaint Narrative Length (Word Count)')
    plt.xlabel('Word Count')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'narrative_length_distribution.png'))
    plt.close()

    # Save summary stats
    stats_file = os.path.join(REPORT_STATS_DIR, 'eda_summary.txt')
    with open(stats_file, 'w') as f:
        f.write("EDA Summary Statistics\n")
        f.write("======================\n\n")
        
        f.write(f"Total Complaints: {len(df)}\n\n")
        
        f.write("Complaints by Product:\n")
        f.write(df['Product'].value_counts().to_string())
        f.write("\n\n")
        
        f.write("Narrative Length Statistics:\n")
        f.write(df['narrative_length'].describe().to_string())
        f.write("\n")

    print(f"EDA completed. Outputs saved to {REPORT_IMAGES_DIR} and {REPORT_STATS_DIR}")

if __name__ == "__main__":
    generate_eda()