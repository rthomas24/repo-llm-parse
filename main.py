import os
import sys
import hashlib
import logging
from pathlib import Path
from dotenv import load_dotenv

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
        'IGNORE_GITIGNORE': os.getenv('IGNORE_GITIGNORE', 'FALSE').upper() == 'TRUE'
    }
    return env_vars

def should_ignore(path, ignore_files, ignore_dirs, git_path, gitignore_spec):
    """Check if the path should be ignored based on ignore lists and .gitignore spec."""
    relative_path = path.relative_to(git_path)
    # Check if the file or directory is in the ignore lists
    if path.is_file() and path.name in ignore_files:
        return True
    if any(part in ignore_dirs for part in relative_path.parts):
        return True
    # Check if the path matches any pattern in .gitignore
    if gitignore_spec and gitignore_spec.match_file(str(relative_path)):
        return True
    return False

def get_file_paths(git_path, ignore_files, ignore_dirs, gitignore_spec):
    """Recursively get all file paths in the git repository, excluding ignored files and directories."""
    files = []
    for root, dirs, filenames in os.walk(git_path):
        root_path = Path(root)
        # Exclude directories that should be ignored
        dirs[:] = [d for d in dirs if not should_ignore(root_path / d, ignore_files, ignore_dirs, git_path, gitignore_spec)]
        for filename in filenames:
            file_path = root_path / filename
            if should_ignore(file_path, ignore_files, ignore_dirs, git_path, gitignore_spec):
                logging.debug(f"Ignoring file: {file_path}")
                continue
            files.append(file_path)
    return files

def write_txt(file_path, skip_empty_files, save_directory, git_path):
    """Read the content of a file and write it to an individual text file with an MD5 hash."""
    try:
        if skip_empty_files and file_path.stat().st_size == 0:
            logging.info(f"Skipping empty file: {file_path}")
            return
        with file_path.open('r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            md5_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            relative_path = file_path.relative_to(git_path)
            save_path = save_directory / f"{relative_path.name}_{md5_hash}.txt"
            save_path.write_text(content, encoding='utf-8')
            logging.info(f"Written to: {save_path}")
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")

def main():
    """Main function to execute the script."""
    setup_logging()
    env = load_environment()
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

    for file_path in files:
        logging.info(f"Processing file: {file_path}")
        write_txt(file_path, skip_empty_files, save_directory, git_path)

    logging.info(f"Training data can be found in '{save_directory}' directory.")

if __name__ == '__main__':
    main()
