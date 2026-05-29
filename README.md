<div align="center">

# Synthetic Data Generator

A conversational LangGraph agent that generates large, realistic, and intentionally messy synthetic datasets ready for machine learning and deep learning workflows.

</div>

---

## What it does

You describe a domain in plain language. The agent asks a few clarifying questions — use case, scale, key features, noise requirements — then designs a schema, writes Python code, and executes it to produce CSV files. The generated data includes realistic noise: missing values, outliers, skewed distributions, mixed date formats, and duplicate rows. It is meant to resemble production data before cleaning, not textbook examples.

---

## How it works

The agent is a LangGraph `StateGraph` with four nodes and two conditional edges.

```
START
  |
  v
gather_requirements  <──────────────────────────────┐
  |                                                  |
  |  (interrupt: pause, ask user one question)       |
  |  (resume: user answers via Command)              |
  |                                                  |
  v  should_gather_more()                            |
  |── "gather" ─────────────────────────────────────┘
  |── "design"
  |
  v
design_schema
  |
  v
write_code  <──────────────────────────────────────┐
  |                                                 |
  v                                                 |
run_code                                            |
  |                                                 |
  v  should_retry()                                 |
  |── "retry"  (error + attempts < 3) ─────────────┘
  |── "success" ──> END
  └── "give_up"  ──> END
```

### Node descriptions

**`gather_requirements`**
Runs an LLM call with a system prompt that instructs Gemini to ask one clarifying question per turn. When it decides enough context has been gathered, it outputs a `%%READY%%` marker. Each question triggers `interrupt(question)`, which pauses the graph and hands control back to `main.py`. The graph resumes when `main.py` calls `app.invoke(Command(resume=answer))`.

**`design_schema`**
Reads the full conversation history and asks Gemini to produce a raw JSON schema — no markdown, no explanation. The schema describes each table with row counts (minimum 5 000), and each column with type, `null_rate`, `outlier_rate`, distribution family, and distribution parameters. A regex + `json.loads` extracts the object; a minimal fallback is used if parsing fails.

**`write_code`**
Passes the schema to Gemini and asks it to write a complete, self-contained Python script. The prompt mandates:

- three noise helper functions (`add_nulls`, `add_outliers`, `add_format_noise`) with exact signatures
- numpy vectorization for all data generation (no Python `for` loops per row)
- column-level distribution: `normal`, `lognormal`, `uniform`, `poisson`, `skewed`
- `null_rate` and `outlier_rate` applied per column from the schema
- `add_format_noise` applied to date and phone columns (~5% mixed formats)
- ~1% duplicate rows injected and reshuffled
- output saved as CSV to `output_path`

On retries the prompt changes: if the previous error was a `SyntaxError` (indicating truncated output), it instructs the LLM to write shorter, more vectorized code. Otherwise it includes the runtime traceback as-is.

**`run_code`**
Compiles the generated code first (`compile(code, "<generated>", "exec")`) to catch syntax errors before execution. A `SyntaxError` is returned immediately with a directive to write more compact code. On success, it reads the output directory and lists the generated CSV files.

### Conditional edges

| Function             | Input checked                         | Routes                                                       |
| -------------------- | ------------------------------------- | ------------------------------------------------------------ |
| `should_gather_more` | `%%READY%%` in last `AIMessage`       | `"gather"` → loop, `"design"` → proceed                      |
| `should_retry`       | `state["error"]`, `state["attempts"]` | `"success"` → END, `"retry"` → write_code, `"give_up"` → END |

---

## LangGraph patterns used

| Pattern                                         | Where in code                                               |
| ----------------------------------------------- | ----------------------------------------------------------- |
| `StateGraph` + `TypedDict` state                | `agent.py` — `State` class                                  |
| `Annotated[list, add_messages]` reducer         | `State.messages` — appends, never overwrites                |
| `interrupt(value)` — pause for human input      | `gather_requirements` node                                  |
| `Command(resume=value)` — resume from interrupt | `main.py` interrupt loop                                    |
| `MemorySaver` checkpointer                      | `build_graph()` — required for `interrupt` to persist state |
| Named conditional edges                         | `should_gather_more`, `should_retry`                        |
| `get_state(config).next`                        | `main.py` — detects whether graph has reached END           |

---

## Generated data characteristics

- **Row count:** minimum 5 000; default 10 000; respects user request
- **Missing values:** 5–15% nulls on non-key columns, injected via `add_nulls`
- **Outliers:** 1–3% of numeric values multiplied by 6×, injected via `add_outliers`
- **Format inconsistencies:** ~5% of date and phone values use alternate formats (`dd/mm/yyyy`, `yyyy/mm/dd`, `mm-dd-yyyy`)
- **Duplicates:** ~1% of rows duplicated and reshuffled back in
- **Distributions:** normal, log-normal, uniform, Poisson, or exponential per column — matched to domain (e.g. salary → log-normal, age → normal, event counts → Poisson)
- **Categorical skew:** `np.random.choice` with non-uniform weights — realistic class imbalance

---

## Project structure

```
langgraph-data-quality-agent/
├── agent.py          # State, nodes, conditional edges, build_graph()
├── main.py           # CLI entry point — interrupt loop driver
├── requirements.txt
├── .env              # GOOGLE_API_KEY (not committed)
├── .env.example
└── output_*/         # Generated at runtime, one timestamped dir per run
```

---

## Setup

**1. Clone and create a virtual environment**

```bash
git clone <repo-url>
cd langgraph-data-quality-agent
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Set your API key**

Copy `.env.example` to `.env` and add your [Google AI Studio](https://aistudio.google.com/) key:

```
GOOGLE_API_KEY=your_key_here
```

---

## Usage

```bash
python main.py
```

The agent starts a conversation:

```
Synthetic Data Generator — LangGraph + Gemini 2.5 Flash
=======================================================
Describe your domain and dataset needs:
> credit card transactions with fraud cases for a bank

What is the primary use case for this dataset?
> training a binary classification model for fraud detection

How many rows do you need?
> 50000

What level of realism do you require — should the data include noise,
missing values, outliers, and class imbalance?
> yes, everything, make it as realistic as possible

[Designing schema...]
[Writing code — attempt 1/3...]
[Executing generated code...]
Saved 50500 rows -> output_credit_card_transactions_.../transactions.csv
Saved 50500 rows -> output_credit_card_transactions_.../cardholders.csv

=======================================================
SUCCESS — 2 CSV file(s) in 'output_credit_card_transactions_20260529_141200':
  cardholders.csv    (4,821.3 KB)
  transactions.csv   (9,204.7 KB)
Attempts: 1
```

---

## Requirements

| Package                         | Purpose                                                |
| ------------------------------- | ------------------------------------------------------ |
| `langgraph>=0.2.0`              | Graph execution, `interrupt`, `MemorySaver`, `Command` |
| `langchain-google-genai>=1.0.0` | Gemini 2.5 Flash via LangChain                         |
| `langchain-core>=0.2.0`         | `HumanMessage`, `AIMessage`, `SystemMessage`           |
| `langchain>=0.2.0`              | LangChain base                                         |
| `faker>=20.0.0`                 | Realistic text data (names, emails, phones, dates)     |
| `pandas>=2.0.0`                 | DataFrame construction and CSV export                  |
| `numpy>=1.24.0`                 | Vectorized data generation and distributions           |
| `python-dotenv>=1.0.0`          | `.env` file loading                                    |
| `typing_extensions>=4.7.0`      | `TypedDict` backport                                   |
