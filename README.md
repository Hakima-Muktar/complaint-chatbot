# Intelligent Complaint Analysis for Financial Services
This project implements a Retrieval-Augmented Generation (RAG) system for intelligent analysis of consumer financial complaints. It leverages the Consumer Financial Protection Bureau (CFPB) dataset to provide a chatbot interface that can understand and answer questions about consumer complaints.

##Project Structure
```
├── data/               # Data directory (raw and processed)
├── notebooks/          # Jupyter notebooks for analysis and experimentation
├── report/             # Generated reports, images, and statistics
├── src/                # Source code directory
│   ├── data_preprocessing.py   # Task 1: Data cleaning and filtering
│   ├── eda_analysis.py         # Task 1: Exploratory Data Analysis
│   ├── sampling.py             # Task 2: Stratified sampling
│   ├── chunking.py             # Task 2: Text chunking
│   ├── embedding.py            # Task 2: Embedding generation
│   └── vector_store_builder.py # Task 2: FAISS vector store creation
├── vector_store/       # Directory for persisted vector stores
└── requirements.txt    # Project dependencies
```
