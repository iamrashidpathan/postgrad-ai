# Developer Tooling Cheatsheet

A step-by-step quick reference for **pip, uv, conda, Git, and Jupyter Notebook magic commands**.

---

## 1. pip — Python Package Management

`pip` is the standard Python package installer.

### Step 1: Install a package

```bash
pip install requests
```

### Step 2: Install a specific package version

```bash
pip install torch==2.1.0
```

### Step 3: Upgrade a package

```bash
pip install --upgrade pip
```

```bash
pip install --upgrade numpy
```

### Step 4: Uninstall a package

```bash
pip uninstall requests
```

### Step 5: List installed packages

```bash
pip list
```

### Step 6: Show package details

```bash
pip show numpy
```

### Step 7: Save installed packages to `requirements.txt`

```bash
pip freeze > requirements.txt
```

### Step 8: Install packages from `requirements.txt`

```bash
pip install -r requirements.txt
```

### Step 9: Install a package directly from a Git repository

```bash
pip install git+https://github.com/user/repo.git
```

### Step 10: Install packages on systems that block global installations

```bash
pip install pandas --break-system-packages
```

### Step 11: Clear pip's download cache

```bash
pip cache purge
```

---

# 2. uv — Fast Package and Project Manager

`uv` is a fast package and project manager for Python environments and dependencies.

## Step 1: Install uv on macOS/Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Step 2: Create a virtual environment

```bash
uv venv
```

## Step 3: Install a package

```bash
uv pip install requests
```

## Step 4: Install packages from a requirements file

```bash
uv pip install -r requirements.txt
```

## Step 5: Generate a locked requirements file

```bash
uv pip compile requirements.in -o requirements.txt
```

## Step 6: Synchronize the environment with a lockfile

```bash
uv pip sync requirements.txt
```

## Step 7: Add a dependency to a uv-managed project

```bash
uv add numpy
```

## Step 8: Remove a dependency

```bash
uv remove numpy
```

## Step 9: Run a script inside the project environment

```bash
uv run train.py
```

## Step 10: Install a specific Python version using uv

```bash
uv python install 3.11
```

---

# 3. conda — Environment and Package Management

`conda` manages environments, packages, and non-Python dependencies.

## Step 1: Create an environment with a specific Python version

```bash
conda create -n genai python=3.10
```

## Step 2: Activate the environment

```bash
conda activate genai
```

## Step 3: Deactivate the current environment

```bash
conda deactivate
```

## Step 4: Install a package

```bash
conda install numpy
```

## Step 5: Remove a package

```bash
conda remove numpy
```

## Step 6: List packages in the active environment

```bash
conda list
```

## Step 7: List all conda environments

```bash
conda env list
```

## Step 8: Export the current environment

```bash
conda env export > environment.yml
```

## Step 9: Create an environment from a YAML file

```bash
conda env create -f environment.yml
```

## Step 10: Delete an environment

```bash
conda env remove -n genai
```

## Step 11: Install a package from a specific channel

```bash
conda install -c conda-forge transformers
```

---

# 4. Git — Version Control

Git tracks code changes and supports collaboration through commits, branches, and remote repositories.

## Step 1: Initialize a new Git repository

```bash
git init
```

## Step 2: Clone a remote repository

```bash
git clone https://github.com/user/repo.git
```

## Step 3: Check the repository status

```bash
git status
```

## Step 4: Stage a specific file

```bash
git add notebook.ipynb
```

## Step 5: Stage all changed files

```bash
git add .
```

## Step 6: Commit staged changes

```bash
git commit -m "Add Week 7 notebook"
```

## Step 7: Create a new branch

```bash
git branch feature-x
```

## Step 8: Switch to another branch

```bash
git switch feature-x
```

Alternative:

```bash
git checkout feature-x
```

## Step 9: Create and switch to a new branch

```bash
git checkout -b feature-x
```

## Step 10: Merge a branch

```bash
git merge feature-x
```

## Step 11: Push commits to a remote repository

```bash
git push origin main
```

## Step 12: Pull changes from a remote repository

```bash
git pull origin main
```

## Step 13: Fetch remote changes without merging

```bash
git fetch origin
```

## Step 14: View configured remote repositories

```bash
git remote -v
```

## Step 15: View commit history

```bash
git log --oneline -5
```

## Step 16: View unstaged changes

```bash
git diff
```

## Step 17: Temporarily save changes

```bash
git stash
```

## Step 18: Restore stashed changes

```bash
git stash pop
```

## Step 19: Discard changes using reset

> **Warning:** This is destructive.

```bash
git reset --hard HEAD~1
```

## Step 20: Safely undo a commit

```bash
git revert a1b2c3d
```

## Step 21: Create a `.gitignore` file

Example:

```gitignore
__pycache__/
*.env
.ipynb_checkpoints/
```

---

# 5. Jupyter Notebook Line Magics (`%`)

Line magics begin with `%` and apply to a single line.

## Step 1: Measure execution time of one line

```python
%time sum(range(10**6))
```

## Step 2: Benchmark a line repeatedly

```python
%timeit sum(range(1000))
```

## Step 3: List variables

```python
%who
```

## Step 4: Display detailed variable information

```python
%whos
```

## Step 5: Show the current working directory

```python
%pwd
```

## Step 6: Change the working directory

```python
%cd /content/drive/MyDrive
```

## Step 7: View environment variables

```python
%env
```

## Step 8: Set an environment variable

```python
%env MY_KEY=value
```

## Step 9: Load a Python script into a cell

```python
%load utils.py
```

## Step 10: Execute an external Python script

```python
%run train.py
```

## Step 11: Display Matplotlib plots inside the notebook

```python
%matplotlib inline
```

## Step 12: View command history

```python
%history -n
```

## Step 13: Reset variables

```python
%reset -f
```

## Step 14: Install a package into the active notebook kernel

```python
%pip install pandas
```

---

# 6. Jupyter Notebook Cell Magics (`%%`)

Cell magics begin with `%%` and apply to the entire cell.

## Step 1: Measure execution time for an entire cell

```python
%%time

total = sum(range(10**7))
```

## Step 2: Benchmark an entire cell repeatedly

```python
%%timeit

sum(range(1000))
```

## Step 3: Write cell contents to a file

```python
%%writefile app.py

print("hello")
```

## Step 4: Run a cell as a Bash script

```bash
%%bash

ls -la
```

## Step 5: Render raw HTML

```html
%%html

<b>Bold text</b>
```

## Step 6: Capture or suppress cell output

```python
%%capture output

print("hidden")
```

## Step 7: Render LaTeX

```latex
%%latex

$E=mc^2$
```

## Step 8: Run SQL queries from a notebook

Requires the `ipython-sql` extension.

```sql
%%sql

SELECT * FROM users;
```

---

# Quick Workflow

## Using pip

```bash
pip install package_name
pip freeze > requirements.txt
pip install -r requirements.txt
```

## Using uv

```bash
uv venv
uv pip install package_name
uv run script.py
```

## Using conda

```bash
conda create -n project python=3.10
conda activate project
conda install package_name
```

## Using Git

```bash
git init
git add .
git commit -m "Initial commit"
git push origin main
```

## Using Jupyter

```python
%timeit expression
%whos
%pwd
%run script.py
%pip install package_name
```

---

## Summary

| Tool | Primary Purpose |
|---|---|
| `pip` | Install and manage Python packages |
| `uv` | Fast package, environment, and project management |
| `conda` | Manage environments, packages, and system dependencies |
| `git` | Version control and collaboration |
| Jupyter `%` magics | Execute single-line notebook commands |
| Jupyter `%%` magics | Execute cell-level commands |
