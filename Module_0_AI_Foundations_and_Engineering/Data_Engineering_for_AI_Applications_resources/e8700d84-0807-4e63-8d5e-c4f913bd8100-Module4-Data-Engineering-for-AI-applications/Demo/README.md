# Data-Centric Thinking for AI System Development — Hands-on Demos

Companion hands-on materials for **Session 4: Data Engineering for AI Applications** —
*"Data-Centric Thinking for AI System Development."*

This package contains four **independent, self-contained demos** that match the session deck:

1. **ETL Demo** — Extract, Transform, Load on a raw CSV
2. **Data Cleaning Pipeline Demo** — a repeatable Profile → Clean → Validate → Log pipeline
3. **Text Preprocessing Demo** — Tokenization → Normalization → Stopword Removal → Stemming vs Lemmatization
4. **SQL Basics Demo** — every core SQL concept from the session, executed against a real SQLite database

None of the demos depend on each other's output — you (or your learners) can run them in any order.

---

## Files in this package

| File | Description |
|---|---|
| `Data_Centric_Thinking_for_AI_System_Development_Demos.ipynb` | The main notebook — all four demos, fully executed with real output |
| `sql_commands.sql` | Standalone reference of every SQL command used in Demo 4, runnable independently against SQLite |
| `requirements.txt` | Python dependencies needed to run the notebook |
| `README.md` | This file |

Running the notebook will also generate, in the same folder:
- `raw_orders.csv`, `clean_orders.csv` — Demo 1 (ETL)
- `customers_raw.csv`, `customers_clean.csv`, `cleaning_change_log.csv` — Demo 2 (Data Cleaning Pipeline)
- `training.db` — a SQLite database used by Demos 1 & 4

---

## Setup

1. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```
   Then open `Data_Centric_Thinking_for_AI_System_Development_Demos.ipynb` and run all cells
   (`Cell > Run All`, or `Kernel > Restart & Run All`).

No internet connection is required — every demo runs fully offline.

---

## What's inside each demo

### Demo 1 — ETL on a Raw CSV
A synthetic, intentionally messy dataset (missing names, inconsistent date formats, invalid amounts,
inconsistent casing) is taken through a full **Extract → Transform → Load** pipeline using `pandas`.
The cleaned result is saved to CSV and loaded into a SQLite table.

### Demo 2 — Data Cleaning Pipeline
A separate customer records dataset (invalid ages, inconsistent city casing, a missing city, mixed date
formats) is run through a **repeatable** four-stage pipeline:
- **Profile** — quantify what's wrong before touching anything
- **Clean** — apply fixes consistently across every affected column
- **Validate** — confirm each fix actually worked
- **Log** — record what changed and why, in an auditable change log

### Demo 3 — Text Preprocessing
Raw customer feedback text is run through a full NLP preprocessing pipeline:
- **Tokenization** — splitting text into word tokens
- **Normalization** — lowercasing and cleaning punctuation/contractions
- **Stopword removal** — filtering out low-signal common words
- **Stemming vs Lemmatization** — a side-by-side comparison table showing how each technique reduces
  words differently (e.g., "replies" → "repli" via stemming vs "reply" via lemmatization)

> **Note:** stopword removal uses a small built-in list, and lemmatization uses a small illustrative
> lookup dictionary, so the whole demo runs **offline** without downloading NLTK's `stopwords` or
> `wordnet` corpora. Stemming uses NLTK's `PorterStemmer`, which is rule-based and needs no download.
> The notebook includes a table mapping each offline stand-in to its production equivalent (e.g., a full
> spaCy or NLTK/WordNet setup).

### Demo 4 — SQL Basics
A small relational schema (`customers` and `sales` tables, ~12 customers and 60 sales records) is created
in SQLite. Every SQL concept from the session is executed live, with real output:
- `SELECT` & `WHERE`
- `JOIN` (with a dedicated INNER vs LEFT comparison)
- `GROUP BY` & aggregation functions (`COUNT`, `SUM`, `AVG`), including a revenue-by-category chart
- `HAVING` vs `WHERE`
- `ORDER BY` & `LIMIT`
- Subqueries
- The session's SQL practice challenge, fully answered and executed

`sql_commands.sql` contains the same queries as a standalone, portable script you can run directly
with any SQLite client, independent of the notebook.

---

## Setting Up SQLite & Running SQL Commands in the Terminal

The notebook creates `training.db` for you automatically, but you can also work with SQLite directly
from the terminal — useful for practicing SQL outside the notebook, or for verifying `sql_commands.sql`
independently.

### 1. Install the SQLite command-line tool

SQLite itself needs no server or account — just a single command-line binary.

**macOS** (via Homebrew):
```bash
brew install sqlite
```

**Windows:**
1. Run command `winget install SQLite.SQLite`
2. Unzip it and add the folder to your system `PATH`
3. Confirm it works: `sqlite3 --version`s

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get update
sudo apt-get install sqlite3
```

Verify the install on any platform:
```bash
sqlite3 --version
```

> **Note:** if you only plan to run SQL through the notebook, no separate install is needed —
> Python's built-in `sqlite3` module (used throughout the notebook) requires no setup at all.

### 2. Create the database and load the schema

From the project folder (run the notebook at least once first, or create a fresh database):

```bash
# Option A: build a fresh database from the reference script
sqlite3 training.db < sql_commands.sql
```

This creates `training.db`, builds the `customers` and `sales` tables, inserts the sample seed rows,
and runs every query in the file in one pass, printing each result to the terminal.

```bash
# Option B: open the database the notebook already created
sqlite3 training.db
```

### 3. Write and run SQL commands interactively

Once inside the `sqlite3` prompt (you'll see `sqlite>`), you can type SQL directly:

```sql
sqlite> SELECT * FROM customers ORDER BY name;
sqlite> SELECT name, city FROM customers WHERE city = 'Mumbai';
sqlite> SELECT COUNT(*) FROM customers;
sqlite> SELECT * FROM customers LIMIT 5;
sqlite> SELECT name, city FROM customers WHERE city = 'Mumbai';
sqlite> SELECT category, SUM(amount) AS revenue
   ...> FROM sales
   ...> GROUP BY category
   ...> ORDER BY revenue DESC;
```

- End every statement with a semicolon (`;`) — SQLite waits for it before running the command.
- Multi-line statements are fine; the prompt changes to `...>` until it sees the closing `;`.
- Press `Ctrl+D` (macOS/Linux) or `Ctrl+Z` then `Enter` (Windows) to exit.

Handy dot-commands (no semicolon needed) for working inside the prompt:

| Command | Purpose |
|---|---|
| `.tables` | List all tables in the database |
| `.schema sales` | Show the `CREATE TABLE` statement for a table |
| `.headers on` | Show column names above query results |
| `.mode column` | Print results in aligned columns |
| `.quit` | Exit the SQLite prompt |

A typical session:
```bash
$ sqlite3 training.db
sqlite> .headers on
sqlite> .mode column
sqlite> .tables
customers  sales
sqlite> SELECT * FROM customers LIMIT 3;
sqlite> .quit
```

### 4. Run a `.sql` file non-interactively (no prompt)

To run `sql_commands.sql` and immediately return to your normal terminal (useful for scripting or CI):
```bash
sqlite3 training.db < sql_commands.sql
```
To save the output to a file instead of printing it to the terminal:
```bash
sqlite3 training.db < sql_commands.sql > sql_output.txt
```

---

## Facilitation tips

- Each demo maps to one hands-on segment in the session agenda — run them live, or have learners
  follow along on their own machines.
- The ETL and Data Cleaning Pipeline demos intentionally use two *different* messy datasets, so learners
  see that the same underlying discipline (profile, fix, validate) applies across different data problems.
- The text preprocessing demo's stemmed vs lemmatized comparison table is a good discussion point —
  ask learners to spot which words changed differently and why.
- The SQL demo's schema (customers + sales) is reused across `SELECT`, `JOIN`, `GROUP BY`, `HAVING`,
  and subqueries, so learners build up query complexity on data they already recognize.
