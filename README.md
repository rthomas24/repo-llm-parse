# repo-llm-parse

## Overview

`repo-llm-parse` scans a given Git project directory, processes files while respecting `.gitignore` rules, and creates a vector store of the project’s content. It leverages OpenAI embeddings for semantic search and allows querying relevant files using natural language prompts. This makes it ideal for developers looking to extract insights from large codebases.

## Key Features

- **Parse Git Repositories**: Automatically reads and respects `.gitignore` rules.
- **OpenAI Integration**: Embeds code content using OpenAI embeddings for semantic similarity.
- **Search & Retrieve**: Quickly search through code with natural language queries.
- **Vector Storage**: Stores document embeddings using FAISS for fast querying.

## Setup and Installation

Follow these steps to set up the repository:

1. **Clone the Repository**:

    ```sh
    git clone <repo-url>
    cd repo-llm-parse
    ```

2. **Install Dependencies**:
    Ensure you have Python installed. Run:

    ```sh
    pip install -r requirements.txt
    ```

3. **Configure Environment Variables**:
    - Copy `.env.sample` to `.env`:

    ```sh
    cp .env.sample .env
    ```

    - Edit `.env` and fill in the following values:
        - `OPENAI_API_KEY`: Your OpenAI API key.
        - `GIT_PROJECT_DIRECTORY`: Path to the Git project.
        - `IGNORE_FILES` and `IGNORE_DIRS`: Files and directories to skip.
        - `SAVE_DIRECTORY`: Directory for storing parsed data.

## How to Run

1. **Navigate to the project directory**:

    ```sh
    cd repo-llm-parse
    ```

2. **Run the script**:

    ```sh
    python main.py
    ```

3. **Provide your query when prompted**:
    - Example:

    ```sh
    Enter your query: Explain how the authentication logic works
    ```

    - The script will retrieve relevant files and generate a detailed response using the OpenAI API.

## Getting the Most Out of It

- **Fine-tune your `.env` settings**: Exclude unnecessary files or directories to speed up processing.
- **Use meaningful queries**: Ask detailed questions for better results.
- **Leverage the vector store**: If the project is large, the vector store allows for faster querying of relevant files.

## Troubleshooting

- **Missing API Key Error**: Ensure your `.env` file has a valid `OPENAI_API_KEY`.
- **Empty Query Error**: Provide a valid query to search the vector store.
- **Dependency Issues**: Install missing dependencies using:

    ```sh
    pip install -r requirements.txt
    ```
