import fetch from 'node-fetch';
import pLimit from 'p-limit';
import dotenv from 'dotenv';
import ignore from 'ignore';
import path from 'path';

dotenv.config();

(async () => {
  const githubUrl = 'https://github.com/rthomas24/gpt-angular-starter-kit';

  try {
    // Extract owner and repo from the GitHub URL
    const match = githubUrl.match(/https:\/\/github\.com\/([^\/]+)\/([^\/]+)/);
    if (!match) {
      throw new Error('Invalid GitHub URL.');
    }

    const owner = match[1];
    const repo = match[2];

    const accessToken = process.env.GITHUB_ACCESS_TOKEN;

    const headers = {
    
    };

    if (accessToken) {
      headers['Authorization'] = `token ${accessToken}`;
    }

    // Fetch repository information to get the default branch
    const repoInfoResponse = await fetch(`https://api.github.com/repos/${owner}/${repo}`, { headers });
    if (!repoInfoResponse.ok) {
      throw new Error('Failed to fetch repository info.');
    }
    const repoInfo = await repoInfoResponse.json();
    const defaultBranch = repoInfo.default_branch;

    // Get the SHA of the default branch
    const refResponse = await fetch(`https://api.github.com/repos/${owner}/${repo}/git/refs/heads/${defaultBranch}`, { headers });
    if (!refResponse.ok) {
      throw new Error('Failed to fetch ref info.');
    }
    const refInfo = await refResponse.json();
    const commitSha = refInfo.object.sha;

    // Fetch the tree recursively
    const treeResponse = await fetch(`https://api.github.com/repos/${owner}/${repo}/git/trees/${commitSha}?recursive=1`, { headers });
    if (!treeResponse.ok) {
      throw new Error('Failed to fetch tree info.');
    }
    const treeInfo = await treeResponse.json();

    if (treeInfo.truncated) {
      console.warn('Warning: The repository tree is truncated because it is too large.');
    }

    let files = treeInfo.tree.filter(item => item.type === 'blob');

    // Check if .gitignore exists in the repository
    const gitignoreEntry = files.find(file => file.path === '.gitignore');

    let ig;

    if (gitignoreEntry) {
      // Fetch the .gitignore content
      const gitignoreResponse = await fetch(`https://raw.githubusercontent.com/${owner}/${repo}/${defaultBranch}/.gitignore`);
      if (!gitignoreResponse.ok) {
        console.warn('Failed to fetch .gitignore file.');
      } else {
        const gitignoreContent = await gitignoreResponse.text();

        ig = ignore();
        ig.add(gitignoreContent);
      }
    } else {
      ig = ignore();
    }

    // Add additional ignore patterns for files and directories
    ig.add([
      'package-lock.json',
      'yarn.lock',
      'pnpm-lock.yaml',
      'composer.lock',
      'Gemfile.lock',
      'node_modules/',
      'dist/',
      'build/',
      'out/',
      'target/',
      '.idea/',
      '.vscode/',
      '.DS_Store',
      'Thumbs.db',
      '*.log',
      '*.tmp',
      '*.cache',
      '*.class',
      '*.pyc',
      '*.o',
      '*.so',
      '*.dll',
      '*.test.*',
      '*.spec.*',
      '*.md',
      '*.markdown',
      '*.exe',
      '*.zip',
      '*.tar',
      '*.gz',
      '*.7z',
      '*.png',
      '*.jpg',
      '*.jpeg',
      '*.gif',
      '*.mp3',
      '*.mp4',
      '*.avi',
      '*.mov',
      '*.wmv',
      '*.pdf',
      '*.doc',
      '*.docx',
      '*.xls',
      '*.xlsx',
      '*.ppt',
      '*.pptx',
      '*.bin',
    ]);

    // List of common binary file extensions
    const binaryExtensionsList = [
      'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'ico',
      'mp3', 'wav', 'flac', 'aac', 'ogg',
      'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv',
      'pdf', 'exe', 'dll', 'so', 'bin', 'dat', 'class', 'jar',
      'zip', 'tar', 'gz', '7z', 'rar', 'iso',
      'eot', 'otf', 'ttf', 'woff', 'woff2',
      'swf', 'psd', 'ai', 'eps', 'sketch',
      'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx',
    ];

    // Create a set of binary extensions for quick lookup
    const binaryExtensionsSet = new Set(binaryExtensionsList);

    // Function to check if a file path is a binary file based on its extension
    function isBinaryFile(filePath) {
      const ext = path.extname(filePath).slice(1).toLowerCase();
      return binaryExtensionsSet.has(ext);
    }

    // Function to determine whether to ignore a file
    function shouldIgnore(filePath) {
      const normalizedPath = filePath.replace(/\\/g, '/');

      if (ig.ignores(normalizedPath)) {
        return true;
      }

      if (isBinaryFile(normalizedPath)) {
        return true;
      }

      return false;
    }

    // Filter out files based on .gitignore patterns and binary file detection
    files = files.filter(file => !shouldIgnore(file.path));

    const limit = pLimit(10);

    // Fetch content for each file with concurrency limit
    const fileContents = await Promise.all(files.map(file => limit(async () => {
      const fileResponse = await fetch(`https://raw.githubusercontent.com/${owner}/${repo}/${defaultBranch}/${file.path}`);
      if (!fileResponse.ok) {
        return null;
      }
      const content = await fileResponse.text();
      return {
        path: file.path,
        content,
      };
    })));

    const result = fileContents.filter(f => f !== null);

    console.log(JSON.stringify(result, null, '\t'));

  } catch (error) {
    console.error('Error loading repository:', error.message);
    process.exit(1);
  }
})();