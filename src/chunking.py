
"""
Text Chunking for Consumer Complaints

This script implements text chunking strategy using LangChain's RecursiveCharacterTextSplitter.
Each chunk is associated with metadata for retrieval purposes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_text_chunks(
    df: pd.DataFrame,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    text_column: str = "Consumer complaint narrative"
) -> List[Dict]:
    """
    Create text chunks from complaint narratives with metadata.
    
    Args:
        df: DataFrame containing complaints
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks
        text_column: Name of the column containing text to chunk
    
    Returns:
        List of dictionaries containing chunks and their metadata
    """
    print(f"Initializing text splitter with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks_data = []
    total_chunks = 0
    
    print(f"\nProcessing {len(df)} complaints...")
    
    for idx, row in df.iterrows():
        if pd.isna(row[text_column]) or str(row[text_column]).strip() == "":
            continue
        
        # Get the complaint text
        complaint_text = str(row[text_column])
        
        # Split into chunks
        text_chunks = text_splitter.split_text(complaint_text)
        
        # Create metadata for each chunk
        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk_data = {
                'text': chunk_text,
                'complaint_id': row.get('Complaint ID', idx),
                'product_category': row.get('Product', ''),
                'product': row.get('Product', ''),
                'issue': row.get('Issue', ''),
                'sub_issue': row.get('Sub-issue', ''),
                'company': row.get('Company', ''),
                'state': row.get('State', ''),
                'date_received': row.get('Date received', ''),
                'chunk_index': chunk_idx,
                'total_chunks': len(text_chunks)
            }
            chunks_data.append(chunk_data)
            total_chunks += 1
        
        # Progress indicator
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} complaints, generated {total_chunks} chunks")
    
    print(f"\nChunking complete!")
    print(f"Total complaints processed: {len(df)}")
    print(f"Total chunks generated: {total_chunks}")
    print(f"Average chunks per complaint: {total_chunks / len(df):.2f}")
    
    return chunks_data


def analyze_chunks(chunks_data: List[Dict]) -> None:
    """
    Analyze the chunking results.
    
    Args:
        chunks_data: List of chunk dictionaries
    """
    print("\n" + "="*80)
    print("CHUNKING ANALYSIS")
    print("="*80)
    
    # Chunk length statistics
    chunk_lengths = [len(chunk['text']) for chunk in chunks_data]
    print(f"\nChunk length statistics:")
    print(f"  Mean: {np.mean(chunk_lengths):.2f} characters")
    print(f"  Median: {np.median(chunk_lengths):.2f} characters")
    print(f"  Min: {np.min(chunk_lengths)} characters")
    print(f"  Max: {np.max(chunk_lengths)} characters")
    print(f"  Std: {np.std(chunk_lengths):.2f} characters")
    
    # Chunks per complaint statistics
    complaint_ids = [chunk['complaint_id'] for chunk in chunks_data]
    unique_complaints = len(set(complaint_ids))
    chunks_per_complaint = len(chunks_data) / unique_complaints
    
    print(f"\nChunks per complaint:")
    print(f"  Average: {chunks_per_complaint:.2f}")
    print(f"  Total unique complaints: {unique_complaints}")
    
    # Product distribution
    products = [chunk['product_category'] for chunk in chunks_data]
    product_counts = pd.Series(products).value_counts()
    print(f"\nChunks per product category:")
    for product, count in product_counts.items():
        print(f"  {product}: {count} ({count/len(chunks_data)*100:.2f}%)")


def save_chunks(chunks_data: List[Dict], output_file: str) -> None:
    """
    Save chunks to a CSV file.
    
    Args:
        chunks_data: List of chunk dictionaries
        output_file: Path to save the chunks
    """
    chunks_df = pd.DataFrame(chunks_data)
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    chunks_df.to_csv(output_file, index=False)
    print(f"\nChunks saved to {output_file}")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")


def main():
    """Main function to create text chunks."""
    # Load the sample data
    input_file = r"C:\Users\user\Desktop\Project\complaint-chatbot\data\processed\complaints.csv"
    output_file =r"C:\Users\user\Desktop\Project\complaint-chatbot\data\processed\sample_chunk.csv"
    
    print(f"Loading sample data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Sample dataset shape: {df.shape}")
    
    # Create chunks
    chunks_data = create_text_chunks(
        df=df,
        chunk_size=500,
        chunk_overlap=50,
        text_column="Consumer complaint narrative"
    )
    
    # Analyze chunks
    analyze_chunks(chunks_data)
    
    # Save chunks
    save_chunks(chunks_data, output_file)


if __name__ == "__main__":
    main()