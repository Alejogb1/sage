import os
import configargparse
import logging
from typing import Optional

from .chat import build_rag_chain, main as chat_main
from .retriever import build_retriever_from_args
from .config import add_all_args

__all__ = ['build_rag_chain', 'sage_index', 'sage_chat']

def sage_index(repo_path: Optional[str] = None, verbose: bool = False):
    """
    Manually index a repository for RAG.
    
    Args:
        repo_path (str, optional): Path to the repository. Defaults to current directory.
        verbose (bool, optional): Enable verbose logging. Defaults to False.
    """
    # Configure logging
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    
    # Use current directory if no path provided
    if repo_path is None:
        repo_path = os.getcwd()
    
    # Create argument parser
    parser = configargparse.ArgumentParser(description='Sage Index CLI')
    add_all_args(parser)
    
    # Set default arguments to use gemini
    parser.set_defaults(
        repo_id=repo_path,
        embedding_provider='gemini',
        llm_retriever=False,
        embedding_model='text-embedding-004',
        llm_provider='gemini',
        llm_model='gemini-1.5-flash',
        vector_store_provider='pinecone',
        retrieval_alpha=0.5,
        retriever_top_k=5
    )
    
    # Parse arguments - this time, allow command-line arguments to override defaults
    args = parser.parse_args()
    
    print("ARGS: ", args)

    # Build retriever (which should index the repository)
    retriever = build_retriever_from_args(args)
    
    print(f"Successfully indexed repository: {repo_path}")
    return retriever

def sage_chat(repo_path: Optional[str] = None, query: Optional[str] = None, 
              model: str = 'gemini', verbose: bool = False):
    """
    Manually chat with a repository.
    
    Args:
        repo_path (str, optional): Path to the repository. Defaults to current directory.
        query (str, optional): Direct query to the repository. If None, starts interactive chat.
        model (str, optional): Language model provider. Defaults to 'gemini'.
        verbose (bool, optional): Enable verbose logging. Defaults to False.
    """
    # Configure logging
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    
    # Use current directory if no path provided
    if repo_path is None:
        repo_path = os.getcwd()
    
    # Create argument parser
    parser = configargparse.ArgumentParser(description='Sage Chat CLI')
    add_all_args(parser)
    
    # Set repo path and model
    parser.set_defaults(
        repo_id=repo_path, 
        llm_provider=model, 
        llm_model='gemini-1.5-flash'  
    )
    
    # Parse arguments
    args = parser.parse_args([])
    
    # Build RAG chain
    rag_chain = build_rag_chain(args)
    
    # If query is provided, run a single query
    if query:
        response = rag_chain.invoke({
            "input": query,
            "chat_history": []
        })
        return response['answer']
    
    # If no query, start interactive chat (similar to main())
    print("Starting interactive chat. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": []
        })
        print("AI:", response['answer'])

if __name__ == "__main__":
    # Run sage_chat when the script is executed directly
    sage_chat('.')