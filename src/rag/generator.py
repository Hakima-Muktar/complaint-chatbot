from typing import Any

def generate_answer(query: str, context: str, llm_model: Any) -> str:
    """
    Generate an answer using the provided LLM model.
    
    Args:
        query: User question.
        context: Retrieved context text.
        llm_model: HuggingFace pipeline or similar callable.
        
    Returns:
        Generated answer string.
    """
    
    prompt = f"""You are a financial analyst assistant for CrediTrust. Answer the question based on the context.

Context:
{context}

Question:
{query}

Answer:"""

    # Check if llm_model is a HF pipeline or callable
    if hasattr(llm_model, '__call__'):
        try:
            # Generate
            # T5 and similar models might need truncation, but pipeline handles some of it.
            # We assume llm_model is configured with reasonable max_length
            response = llm_model(prompt)
            
            # Parse response
            if isinstance(response, list) and len(response) > 0 and 'generated_text' in response[0]:
                return response[0]['generated_text']
            elif isinstance(response, str):
                return response
            else:
                return str(response)
                
        except Exception as e:
            return f"Error: {str(e)}"
            
    return "Error: LLM model not callable"