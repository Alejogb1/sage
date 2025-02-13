import sys
import os

# Ensure the sage package is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    import configargparse
    from sage import sage_index, sage_chat
    from sage.config import add_all_args
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have installed all dependencies:")
    print("1. Run 'pip install -e .' in the project directory")
    print("2. Check that all required packages are installed")
    sys.exit(1)

def main():
    """Main entry point for the Sage CLI."""
    # Create argument parser
    parser = configargparse.ArgumentParser(description='Sage Indexing and Chat CLI')
    
    # Add repository ID as a positional argument BEFORE adding other arguments
    parser.add_argument('repo_id', type=str, nargs='?', default="Alejogb1/GiddySelfassuredConnections", 
                        help='The ID of the repository to index and chat with')
    
    # Add all other arguments
    validator = add_all_args(parser)
    
    # Parse arguments
    print("Parsing arguments...")
    print("sys.argv:", sys.argv)
    
    # Detailed argument parsing debugging
    try:
        args = parser.parse_args()
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        print("Parser configuration:")
        parser.print_help()
        raise
    
    # Validate arguments
    args = validator(args)
    
    try:
        # Index the repository
        print(f"Indexing repository: {args.repo_id}")
        retriever = sage_index(args.repo_id, verbose=args.verbose)

        # Start chat with the repository
        print("\nStarting interactive chat. Type your questions about the repository.")
        sage_chat(args.repo_id, model="gemini")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Ensure you have the latest version of the package")
        print("2. Check your API keys and internet connection")
        print("3. Verify the repository is accessible")

if __name__ == "__main__":
    main()
