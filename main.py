import os
import sys
import hashlib
import logging
from typing import List
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

try:
    import pathspec  # For parsing .gitignore files
except ImportError:
    pathspec = None
    logging.error("Error: 'pathspec' module is required to parse the .gitignore file. Install it using 'pip install pathspec'.")
    sys.exit(1)

def setup_logging():
    """Configure the logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def load_environment():
    """Load environment variables from the .env file."""
    load_dotenv()
    env_vars = {
        'GIT_PROJECT_DIRECTORY': os.getenv('GIT_PROJECT_DIRECTORY'),
        'IGNORE_FILES': os.getenv('IGNORE_FILES', '').split(',') if os.getenv('IGNORE_FILES') else [],
        'IGNORE_DIRS': os.getenv('IGNORE_DIRS', '').split(',') if os.getenv('IGNORE_DIRS') else [],
        'SAVE_DIRECTORY': os.getenv('SAVE_DIRECTORY', 'training_data'),
        'SKIP_EMPTY_FILES': os.getenv('SKIP_EMPTY_FILES', 'TRUE').upper() == 'TRUE',
        'IGNORE_GITIGNORE': os.getenv('IGNORE_GITIGNORE', 'FALSE').upper() == 'TRUE',
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY')
    }
    return env_vars

def should_ignore(path, ignore_files, ignore_dirs, git_path, gitignore_spec):
    """Check if the path should be ignored based on ignore lists and .gitignore spec."""
    relative_path = path.relative_to(git_path)
    if path.is_file() and path.name in ignore_files:
        return True
    if any(part in ignore_dirs for part in relative_path.parts):
        return True
    if gitignore_spec and gitignore_spec.match_file(str(relative_path)):
        return True
    return False

def get_file_paths(git_path, ignore_files, ignore_dirs, gitignore_spec):
    """Recursively get all file paths in the git repository, excluding ignored files and directories."""
    files = []
    for root, dirs, filenames in os.walk(git_path):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if not should_ignore(root_path / d, ignore_files, ignore_dirs, git_path, gitignore_spec)]
        for filename in filenames:
            file_path = root_path / filename
            if should_ignore(file_path, ignore_files, ignore_dirs, git_path, gitignore_spec):
                logging.debug(f"Ignoring file: {file_path}")
                continue
            files.append(file_path)
    return files

def process_file(file_path, skip_empty_files, save_directory, git_path):
    """Read the content of a file and create a Document with its content and metadata."""
    try:
        if skip_empty_files and file_path.stat().st_size == 0:
            logging.info(f"Skipping empty file: {file_path}")
            return None

        with file_path.open('r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        relative_path = file_path.relative_to(git_path)
        md5_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        metadata = {
            'source': str(file_path),
            'relative_path': str(relative_path),
            'file_name': file_path.name,
            'file_type': file_path.suffix,
            'md5_hash': md5_hash,
            'file_size': file_path.stat().st_size
        }

        doc = Document(
            page_content=content,
            metadata=metadata
        )

        save_path = save_directory / f"{relative_path.name}_{md5_hash}.txt"
        save_path.write_text(content, encoding='utf-8')
        logging.info(f"Written to: {save_path}")

        return doc

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None

def create_vector_store(documents: List[Document], embeddings: Embeddings, save_directory: Path) -> FAISS:
    """Create a FAISS vector store from the documents."""
    try:
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )
        
        # Save the vector store
        vector_store_path = save_directory / "vector_store"
        vector_store_path.mkdir(exist_ok=True)
        vector_store.save_local(str(vector_store_path), index_name="code_index")
        logging.info(f"Vector store saved to {vector_store_path}")
        
        return vector_store
    
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        raise

def generate_answer(query: str, retrieved_docs: List[Document], openai_api_key: str) -> str:
    """
    Generate an answer using OpenAI's chat model based on the retrieved documents.

    Args:
        query (str): The user's query.
        retrieved_docs (List[Document]): The documents retrieved from the vector store.
        openai_api_key (str): OpenAI API key.

    Returns:
        str: The generated answer.
    """
    try:
        # Initialize the chat model
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-4o",
            temperature=0.2
        )

        # Create a chat prompt template using ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Use the following pieces of context to answer the question at the end. Always provide a detailed code snippet along with your answer to help improve the user's code. Ensure the code snippet is relevant to the context and addresses the user's question directly. If you don't know the answer, just say that you don't know. Do not make up an answer."
                ),
                (
                    "human",
                    "Context:\n{context}\n\nQuestion: {question}\n\nPlease provide a detailed code snippet that addresses the question and improves the code."
                ),
            ]
        )

        # Create the chain by piping the prompt and the LLM
        chain = prompt | llm

        # Combine the content of retrieved documents as context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Generate the answer by invoking the chain
        answer = chain.invoke(
            {
                "context": context,
                "question": query,
            }
        )

        return answer

    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        return "Sorry, I couldn't generate an answer at this time."

def query_vector_store(vector_store: FAISS, openai_api_key: str):
    """Prompt the user for a query, perform similarity search, and display results using OpenAI's chat model."""
    try:
        query = input("Enter your query: ").strip()
        if not query:
            logging.info("No query entered. Exiting.")
            return
        
        logging.info(f"Performing similarity search for query: '{query}'")
        results = vector_store.similarity_search(query, k=5)
        
        if not results:
            logging.info("No relevant documents found.")
            return
        
        logging.info("Generating answer using OpenAI's chat model...")
        answer = generate_answer(query, results, openai_api_key)
        
        print("\nAnswer:")
        print(answer)
        
    except KeyboardInterrupt:
        logging.info("\nQuery cancelled by user.")
    except Exception as e:
        logging.error(f"Error during querying: {e}")

def main():
    """Main function to execute the script."""
    setup_logging()
    env = load_environment()
    
    # Check for OpenAI API key
    if not env['OPENAI_API_KEY']:
        logging.error("OPENAI_API_KEY is not set in the .env file.")
        sys.exit(1)
    
    git_project_directory = env['GIT_PROJECT_DIRECTORY']
    ignore_files = env['IGNORE_FILES']
    ignore_dirs = env['IGNORE_DIRS']
    save_directory = Path(env['SAVE_DIRECTORY'])
    skip_empty_files = env['SKIP_EMPTY_FILES']
    ignore_gitignore = env['IGNORE_GITIGNORE']

    if not git_project_directory:
        logging.error("GIT_PROJECT_DIRECTORY is not set in the .env file.")
        sys.exit(1)

    git_path = Path(git_project_directory)
    if not git_path.is_dir():
        logging.error(f"GIT_PROJECT_DIRECTORY '{git_project_directory}' not found or not a directory.")
        sys.exit(1)

    if not save_directory.exists():
        save_directory.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created SAVE_DIRECTORY at {save_directory}")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    logging.info("Initialized OpenAI embeddings")

    # Read .gitignore file in git_path
    gitignore_spec = None
    if not ignore_gitignore:
        gitignore_path = git_path / '.gitignore'
        if gitignore_path.exists():
            with gitignore_path.open('r') as f:
                gitignore_patterns = f.read().splitlines()
            gitignore_spec = pathspec.PathSpec.from_lines('gitwildmatch', gitignore_patterns)
            logging.info(f"Loaded .gitignore from {gitignore_path}")
        else:
            logging.warning(f"No .gitignore file found in {git_path}")
    else:
        logging.info("Ignoring .gitignore file as per IGNORE_GITIGNORE setting.")

    files = get_file_paths(git_path, ignore_files, ignore_dirs, gitignore_spec)
    if not files:
        logging.error(f"No files found in git directory: {git_project_directory}")
        sys.exit(1)
    logging.info(f"Found {len(files)} files to process.")

    documents = []
    for file_path in files:
        logging.info(f"Processing file: {file_path}")
        doc = process_file(file_path, skip_empty_files, save_directory, git_path)
        if doc:
            documents.append(doc)

    logging.info(f"Created {len(documents)} documents.")
    
    # Create and save vector store
    vector_store = create_vector_store(documents, embeddings, save_directory)
    logging.info(f"Vector store created with {len(documents)} documents")
    
    # Set up retriever for user queries
    query_vector_store(vector_store, env['OPENAI_API_KEY'])

    logging.info("Script execution completed.")

if __name__ == '__main__':
    main()
