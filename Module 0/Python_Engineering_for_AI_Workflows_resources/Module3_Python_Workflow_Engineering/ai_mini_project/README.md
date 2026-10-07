# AI Mini Project — Standard Project Structure Demo

This is a small, fully working project built to demonstrate the standard
folder structure used by professional AI/ML teams, and how the Python
engineering practices from the session (functions, classes, file handling,
JSON/YAML, exceptions, logging, configuration, type hints, clean code, and
reproducibility) all fit into it.

Nothing here is a toy example with fake code — every file actually runs.
Follow along by opening files in the order described below.

## Project Structure

```
ai_mini_project/
├── data/               raw dataset(s)
├── models/             saved model artifacts (created by train.py)
├── notebooks/           exploration — messy is fine here, by design
├── src/                 production-ready code
│   ├── __init__.py
│   ├── data_loader.py    file handling + exception handling
│   ├── model.py          classes, inheritance, @dataclass
│   ├── utils.py          logging setup
│   └── train.py          ties everything together, saves the model
├── main.py               FastAPI app that serves the trained model
├── configs/
│   └── config.yaml       non-secret settings
├── logs/                 log files and run metrics (created when you run train.py)
├── .env                  secrets — never committed to git (demo value only)
├── .env.example          template for .env — safe to commit
├── .gitignore
├── requirements.txt
└── README.md             this file
```

## Why each folder exists

| Folder / File | Purpose |
|---|---|
| `data/` | Raw and processed datasets. Kept out of git via `.gitignore` — data doesn't belong in version control. |
| `models/` | Where `train.py` saves the trained model (`model.joblib`), so it can be reloaded later without retraining. |
| `notebooks/` | Exploration only. `01_data_exploration.ipynb` looks at the dataset before any training happens — nothing in here is reused directly, it's just for understanding the data. |
| `src/` | The actual reusable code: typed functions and classes, meant to be imported, not copy-pasted. |
| `main.py` | A FastAPI app that loads the model saved in `models/` and serves it over HTTP. |
| `configs/config.yaml` | Every non-secret setting (file paths, model hyperparameters) lives here instead of being hardcoded in `src/`. |
| `logs/` | Where `train.py` and `main.py` write their log files, and where `train.py` saves a `metrics.json` summary after each run. |
| `.env` / `.env.example` | `.env` holds secrets (here, a fake demo API key, used both for logging and to protect the `/predict` endpoint) and is never committed. `.env.example` documents which variables are needed, with placeholder values, and is safe to commit. |

## What each file in `src/` demonstrates

- **`data_loader.py`** — reading a CSV with `pathlib`, and raising clear
  exceptions (`FileNotFoundError`, a custom `DataValidationError`) when
  something is wrong with the input data.
- **`model.py`** — a `BaseModel` class with a `RandomForestModel` subclass
  (inheritance), plus a `TrainingConfig` `@dataclass` for typed settings.
- **`utils.py`** — a single shared `setup_logging()` function, so every
  script in the project logs the same way.
- **`train.py`** — the main script. Loads `configs/config.yaml` and `.env`,
  loads the data, trains the model, handles errors, logs progress, saves
  the trained model to `models/model.joblib` with `joblib`, and saves the
  result as `logs/metrics.json`. Setting `random_state` from the config
  also makes the run reproducible — run it as many times as you like and
  you'll get the same accuracy back.

## Getting Started

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Explore the data (optional)

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 3. Run the training script

From the project root:

```bash
python src/train.py
```

You should see:

```
Training accuracy: 1.0000
```

Afterward, check:
- `logs/train.log` — a timestamped record of the run
- `logs/metrics.json` — the accuracy and random seed used, saved as JSON
- `models/model.joblib` — the trained model, saved with `joblib`

### 4. Serve the model with FastAPI

Once you've trained a model (step 3), start the API:

```bash
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000/docs** for interactive, browser-based API
documentation — you can try every endpoint from there directly.

Or use `curl`:

```bash
# No authentication required
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info

# /predict requires the X-API-Key header (see .env)
curl -X POST http://127.0.0.1:8000/predict \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: sk-demo-12345" \\
  -d '{"features": [0.5, -1.2, 0.3, 1.1, -0.4]}'
```

`GET /model-info` tells you exactly how many features are expected, and in
what order, so you always know what to send to `/predict`. Try the request
above without the `X-API-Key` header, or with the wrong value, and notice
it's rejected (422 if the header is missing entirely, 401 if it's wrong) —
this is the same `.env` pattern from training, now protecting an endpoint.

### 5. Format and lint the code (optional)

This is the same workflow used before committing code in a real project:

```bash
black main.py src/
ruff check main.py src/
```

Both should report no issues — this codebase is already formatted and
linted clean.

## Try It Yourself

A few small changes to see how the pieces fit together:

- Change `n_estimators` in `configs/config.yaml` and re-run `train.py` —
  notice you didn't have to touch any code.
- Delete `data/train.csv` and re-run `train.py` — notice the clear
  `FileNotFoundError` message instead of a confusing crash.
- Open `logs/train.log` after a few runs — notice every run is recorded
  with a timestamp.
- Add a new field to `.env` (and to `.env.example` as a placeholder) and
  read it in `train.py` with `os.getenv(...)`.
- Change the `API_KEY` value in `.env`, restart the API, and try your old
  `curl` command again — notice it's now rejected until you update the
  header to match.

## Notes

- The dataset in `data/train.csv` is synthetic (generated with
  `sklearn.datasets.make_classification`), so results are for
  demonstration only.
- The `.env` value in this demo is a fake placeholder — never commit a
  real API key or credential, even in a personal project.
- If you retrain with different data or settings, restart the API so it
  picks up the new `models/model.joblib` — it's only loaded once, at
  startup.
