import gradio as gr
from src.rag.pipeline import RAGSystem
import os
import json
from datetime import datetime

# Initialize RAG System
# Use absolute path to ensure it works regardless of where the script is run from
base_dir = os.path.dirname(os.path.abspath(__file__))
vector_store_path = os.path.join(base_dir, "vector_store", "faiss_index")

# Check if vector store exists
if not os.path.exists(vector_store_path):
    raise FileNotFoundError(
        f"Vector store not found at {vector_store_path}. "
        "Please run the Task 2 notebook to create the vector store first."
    )

print("Initializing RAG System...")
rag_system = RAGSystem(vector_store_path)
print("RAG System ready!")

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

def log_interaction(query, response):
    """Log user interactions for analysis."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": response['answer'],
        "sources": [s.get('complaint_id', 'N/A') for s in response['context'][:3]]
    }
    with open("logs/chat_interactions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def process_query(message, history):
    """
    Process user message and return updated history.
    """
    if not message or message.strip() == "":
        return "", history
    
    # Append user message to history
    history.append({"role": "user", "content": message})
    
    try:
        # Query RAG system
        response = rag_system.query(message)
        
        # Log the interaction
        log_interaction(message, response)
        
        # Format response with sources
        answer = response['answer']
        sources = response['context'][:3]  # Top 3 sources
        
        # Build formatted response
        formatted_response = f"{answer}\n\n**📚 Sources:**\n"
        for i, src in enumerate(sources, 1):
            company = src.get('company', 'Unknown')
            product = src.get('product', 'N/A')
            text_preview = src.get('text', '')[:150]
            formatted_response += f"\n{i}. **{company}** - {product}\n"
            formatted_response += f"   _{text_preview}..._\n"
            
        history.append({"role": "assistant", "content": formatted_response})
        return "", history
    
    except Exception as e:
        error_msg = f"❌ Error processing your query: {str(e)}\n\nPlease try again or rephrase your question."
        history.append({"role": "assistant", "content": error_msg})
        return "", history

def clear_history():
    """Clear the chat history."""
    return [], ""

# Create Gradio Blocks Interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 💬 CrediTrust Complaint Analysis Assistant
        
        Ask questions about consumer complaints related to:
        - 💳 Credit cards
        - 💰 Personal loans
        - 🏦 Savings accounts
        - 💸 Money transfers
        
        This AI assistant uses a RAG (Retrieval-Augmented Generation) system to provide answers based on real consumer complaints from the CFPB database.
        """
    )
    
    chatbot = gr.Chatbot(height=500, label="Conversation History")
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your question here (e.g., 'Why was my loan denied?')...",
            scale=4,
            label="Your Question"
        )
        submit_btn = gr.Button("Submit", variant="primary", scale=1)
    
    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear Chat & Reset", variant="secondary")
        
    gr.Examples(
        examples=[
            "Why was my loan denied?",
            "How do I dispute a charge on my credit card?",
            "What are common issues with savings accounts?",
            "How long does a money transfer take?",
            "What should I do if my credit card was charged incorrectly?",
            "Why is my savings account showing unexpected fees?"
        ],
        inputs=msg,
        label="Example Queries (Click to try)"
    )

    # Event handlers
    submit_btn.click(
        fn=process_query,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    
    msg.submit(
        fn=process_query,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    
    clear_btn.click(
        fn=clear_history,
        inputs=[],
        outputs=[chatbot, msg]
    )

if __name__ == "__main__":
    demo.launch(
        share=False, 
        server_name="0.0.0.0", 
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft()
    )