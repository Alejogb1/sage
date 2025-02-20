import os

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


def build_llm_via_langchain(provider: str, model: str):
    """Builds a language model via LangChain."""
    # Define the API key requirements for different LLM providers
    provider_key_map = {
        'openai': 'OPENAI_API_KEY',
        'gemini': 'GOOGLE_API_KEY',
        'ollama': None  # Ollama doesn't require an API key
    }

    # Check if the LLM provider is supported
    if provider not in provider_key_map:
        raise ValueError(f"Unrecognized LLM provider {provider}. Contributions are welcome!")

    # Check for required API key
    api_key = provider_key_map.get(provider)
    if api_key and not os.getenv(api_key):
        raise ValueError(f"Please set the {api_key} environment variable to use {provider} LLM.")

    # Build the appropriate LLM based on the provider
    if provider == "openai":
        return ChatOpenAI(model=model or "gpt-4")
    elif provider == "ollama":
        return ChatOllama(model=model or "llama3.1")
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(model=model or "gemini-2.0-flash-exp")
    
    # This line should never be reached due to the earlier check
    raise ValueError(f"Unrecognized LLM provider {provider}. Contributions are welcome!")
