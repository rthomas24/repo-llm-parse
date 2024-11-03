# repo-llm-parse

## Early Development Notice

`repo-llm-parse` is currently in early development. The end goal is to provide support for both local and remote Git repositories using JavaScript and Python. The aim is to make it very easy to interact with a language model (LLM) with your repository's context always attached, enabling it to answer questions and provide better code insights.

## Overview

`repo-llm-parse` enables developers to parse Git repositories, process code files, respect `.gitignore` rules, and create embeddings using OpenAI embeddings. It supports local (Python) and remote GitHub repositories (JavaScript) for seamless integration with OpenAI’s models. This tool is ideal for querying large codebases, retrieving code snippets, and generating code insights.

## Key Features

- **Parse Local & Remote Repositories**: Supports parsing of local repositories (Python) and remote GitHub repositories (JavaScript).
- **.gitignore Compliance**: Automatically respects `.gitignore` rules.
- **OpenAI Integration**: Embeds code and generates answers using OpenAI’s chat models.
- **Natural Language Query**: Query codebases to retrieve relevant snippets.
- **Efficient Storage**: Stores embeddings with FAISS for fast querying (Python).
- **Token Management**: Filters large documents to stay within token limits.

## Components

### Python
Processes a local Git repo, creates code embeddings, and allows natural language querying.

### JavaScript
Fetches a GitHub repo’s contents, respects ignore patterns, and outputs as JSON for further processing.

## Setup and Installation

### Prerequisites
- **Python** 3.7+, **Node.js** 20+, OpenAI API key, GitHub token (optional for private repos).

### Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd repo-llm-parse
   ```

2. Set up Python:
   ```bash
   pip install -r python/requirements.txt
   cp python/.env.sample python/.env
   ```

3. Set up JavaScript:
   ```bash
   cd javascript
   npm install
   cp .env.sample .env
   ```

## Usage

### Python Script

1. Navigate to python directory:
   ```bash
   cd python
   ```

2. Run the script and enter query when prompted:
   ```bash
   python main.py
   ```

### JavaScript Script

1. Set githubUrl in fetchRepoFiles.js.
2. Run the script:
   ```bash
   node fetchRepoFiles.js > output.json
   ```

## Troubleshooting

- Ensure API keys are valid.
- For rate limits, use a GitHub token.
- Install missing dependencies as needed.
