# AI Engineering Environment Setup — Session Files

This folder contains the setup files for the **AI Engineering Environment Setup** session (Foundation Bridge, upGrad Led). Follow the steps below to get your environment ready before the session.

---

## 1. Prerequisites

- Python 3.9 or higher installed
- `pip` (comes bundled with Python)
- Optional but recommended: [Git](https://git-scm.com/) and [Google Colab](https://colab.research.google.com/) account for cloud/GPU access

---

## 2. Check Your Python Version

Open a terminal and run:

```bash
python --version
# or, on some systems:
python3 --version
```

You should see something like `Python 3.10.12`. This project requires **Python 3.9+**.

---

## 3. Set Up a Virtual Environment

Creating an isolated environment keeps this project's packages separate from everything else on your machine.

### Using `venv` (built into Python)

```bash
# Create the environment
python -m venv env

# Activate it
# On macOS/Linux:
source env/bin/activate
# On Windows:
env\Scripts\activate

# Confirm you're inside the environment
which python      # macOS/Linux
where python       # Windows
```

### Using `conda` (alternative)

```bash
conda create -n ai-env python=3.10
conda activate ai-env
```

---

## 4. Install Dependencies

With your virtual environment activated, install everything from `requirements.txt`:

```bash
pip install -r requirements.txt
```

If your system blocks global installs, use:

```bash
pip install -r requirements.txt --break-system-packages
```

---

## 5. How to Change Your Python Version

If your installed Python version doesn't meet the 3.9+ requirement, you have a few options:

### Option A — Install a new Python version directly
Download and install the required version from [python.org/downloads](https://www.python.org/downloads/), then create your virtual environment using that specific version:

```bash
# Example: using a specific Python binary to create the venv
python3.11 -m venv env
```
``` py --version```

``` py --versions```

### Option B — Use `conda` to manage versions
```bash
conda create -n ai-env python=3.11
conda activate ai-env
```

### Option C — Use `uv` (fast, modern option)
```bash
uv python install 3.11
uv venv --python 3.11
```

After switching, re-run `python --version` inside the activated environment to confirm the change took effect.

---

## 6. Run Jupyter Notebook

With your environment activated and dependencies installed, launch Jupyter:

```bash
jupyter notebook
```

This opens Jupyter in your browser. From there, open `AI_Environment_Setup_Hands_On.ipynb` and run the cells from top to bottom.

**Alternative — JupyterLab interface:**

```bash
jupyter lab
```

**Alternative — Google Colab:**
If you'd rather skip local setup entirely, upload the notebook to [Google Colab](https://colab.research.google.com/), which comes with Python, Jupyter, and free GPU access pre-configured. Just run:

```python
!pip install -r requirements.txt
```
in the first cell if you need any packages not already available.

---

## 7. Quick Troubleshooting

| Issue | Fix |
|---|---|
| `python: command not found` | Try `python3` instead, or reinstall Python and ensure it's added to PATH |
| Wrong Python version picked up | Deactivate/reactivate your virtual environment, or specify the binary explicitly (e.g., `python3.11 -m venv env`) |
| `pip install` blocked by system | Add `--break-system-packages`, or make sure your virtual environment is activated first |
| Jupyter can't find installed packages | Confirm the notebook's kernel matches your virtual environment (`Kernel → Change Kernel`) |

---

## Files in This Folder

- `requirements.txt` — all Python packages needed for this session
- `README.md` — this setup guide
- `AI_Environment_Setup_Hands_On.ipynb` — hands-on notebook used during the session

## Remove .venv
```Remove-Item -Recurse -Force .venv```