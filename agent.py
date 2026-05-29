import os
import json
import re
import traceback
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

load_dotenv()

MAX_RETRIES = 3

GATHER_SYSTEM = """You are a synthetic data expert helping a user generate a realistic dataset.

Your job: ask focused clarifying questions ONE AT A TIME to understand their needs fully.

Cover these topics across 2–4 questions:
1. Use case — what will this data be used for? (ML training, DL, analytics, testing?)
2. Scale — how many rows do they need per table? (default: 10 000)
3. Key features — what specific columns or domain entities matter most?
4. Realism requirements — should the data include noise, missing values, outliers, class imbalance?

Rules:
- Ask only ONE question per turn. Keep it short and concrete.
- After you have covered the essentials (≥2 questions answered), output ONLY this exact token on a line by itself:
  %%READY%%
- Never output %%READY%% on the same turn you ask a question."""

SCHEMA_SYSTEM = """You are a data architect. Based on the conversation below, design a detailed schema
for synthetic data generation.

Output ONLY a raw JSON object — no markdown fences, no explanation.

Required format:
{
  "tables": [
    {
      "name": "table_name",
      "rows": 10000,
      "columns": [
        {
          "name": "col_name",
          "type": "uuid|int|float|str|bool|email|name|phone|date",
          "description": "what this field represents",
          "null_rate": 0.05,
          "outlier_rate": 0.02,
          "distribution": "normal|lognormal|uniform|poisson|skewed",
          "params": {"mean": 0, "std": 1}
        }
      ]
    }
  ]
}

Rules:
- rows: honour user's request; default 10 000; never less than 5 000
- null_rate: 0.0 for PKs/required fields, 0.05–0.15 for optional fields
- outlier_rate: 0.01–0.03 for numeric columns
- distribution + params: only for numeric/date columns; skip for str/bool/uuid/email/name/phone
- Include realistic domain-specific categorical values in "description"
- Add foreign-key columns where tables are relational"""

CODE_SYSTEM = """You are an expert Python developer. Output ONLY Python source code — no markdown, no explanation."""


def _llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=32768,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    schema: dict
    code: str
    output_path: str
    error: str
    attempts: int


# ── Node 1: gather_requirements ──────────────────────────────────────────────

def gather_requirements(state: State) -> dict:
    response = _llm().invoke([SystemMessage(content=GATHER_SYSTEM)] + state["messages"])
    content = response.content.strip()

    if "%%READY%%" in content:
        return {"messages": [AIMessage(content=content)]}

    # Pause the graph and wait for the user's answer
    user_answer = interrupt(content)

    return {
        "messages": [AIMessage(content=content), HumanMessage(content=user_answer)],
    }


def should_gather_more(state: State) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and "%%READY%%" in msg.content:
            return "design"
    return "gather"


# ── Node 2: design_schema ────────────────────────────────────────────────────

def design_schema(state: State) -> dict:
    print("\n[Designing schema...]")
    response = _llm().invoke([SystemMessage(content=SCHEMA_SYSTEM)] + state["messages"])
    raw = response.content

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            schema = json.loads(match.group())
        except json.JSONDecodeError:
            schema = _fallback_schema()
    else:
        schema = _fallback_schema()

    return {"messages": [AIMessage(content=raw)], "schema": schema}


def _fallback_schema() -> dict:
    return {"tables": [{"name": "data", "rows": 10000, "columns": [
        {"name": "id", "type": "uuid", "description": "primary key", "null_rate": 0.0, "outlier_rate": 0.0},
        {"name": "value", "type": "float", "description": "generic value", "null_rate": 0.05,
         "outlier_rate": 0.02, "distribution": "normal", "params": {"mean": 50, "std": 10}},
    ]}]}


# ── Node 3: write_code ───────────────────────────────────────────────────────

def write_code(state: State) -> dict:
    attempt_num = state.get("attempts", 0) + 1
    print(f"[Writing code — attempt {attempt_num}/{MAX_RETRIES}...]")

    retry_block = ""
    if state.get("error"):
        if "SyntaxError" in state["error"] or "truncated" in state["error"]:
            retry_block = f"""
CRITICAL — previous code had a syntax error (output was likely truncated):
{state["error"][:300]}

WRITE SHORTER CODE THIS TIME:
- Build each column as a separate numpy array, then assign: df = pd.DataFrame(); df['col'] = arr
- No inline pd.DataFrame({{'col': [...long list...]}}) constructors
- No comments in the code
- No Python for-loops for data generation — use numpy vectorization only
- Each table's generation block must fit in under 25 lines
"""
        else:
            retry_block = f"""
Previous attempt failed — fix this error:
{state["error"][:500]}

Common fixes:
- faker: uuid4(), email(), name(), phone_number(), date_between(start_date='-2y', end_date='today').isoformat()
- numpy: np.random.normal(mean, std, n) not np.random.normal(n)
- np.random.choice weights must sum to 1.0
- output_path must be a string defined at the top of the script
"""

    conversation_summary = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
        for m in state["messages"]
        if isinstance(m, (HumanMessage, AIMessage)) and "%%READY%%" not in m.content
    )

    prompt = f"""Write a complete Python script that generates a realistic, messy synthetic dataset.

Requirements conversation:
{conversation_summary}

Schema to implement:
{json.dumps(state["schema"], indent=2)}

Script requirements:
1. First line: output_path = "{state["output_path"]}"
2. Imports: faker, pandas as pd, numpy as np, os, random, math
3. Seeds: Faker.seed(42); np.random.seed(42); random.seed(42)
4. fake = Faker()

5. Implement these noise helpers EXACTLY:
   def add_nulls(series, rate):
       mask = np.random.random(len(series)) < rate
       return series.where(~mask, other=None)

   def add_outliers(arr, rate, factor=6):
       mask = np.random.random(len(arr)) < rate
       arr = arr.copy().astype(float)
       arr[mask] *= factor
       return arr

   def add_format_noise(values, fmt="date"):
       result = list(values)
       for i in np.where(np.random.random(len(values)) < 0.05)[0]:
           if fmt == "date" and result[i]:
               try:
                   from datetime import datetime
                   d = datetime.fromisoformat(str(result[i]))
                   result[i] = d.strftime(random.choice(["%d/%m/%Y", "%Y/%m/%d", "%m-%d-%Y"]))
               except Exception:
                   pass
       return result

6. Generate data using schema's distribution params:
   - "normal" → np.random.normal(mean, std, n).clip(min_val, max_val)
   - "lognormal" → np.random.lognormal(mean, std, n)
   - "uniform" → np.random.uniform(low, high, n)
   - "poisson" → np.random.poisson(lam, n)
   - "skewed" → np.random.exponential(scale, n)

7. For categorical/str columns: use np.random.choice with skewed weights (not uniform)
8. Apply add_nulls and add_outliers per column using schema null_rate / outlier_rate
9. Apply add_format_noise to date and phone columns
10. Inject ~1% duplicates per table:
    dups = df.sample(frac=0.01, random_state=1)
    df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

11. os.makedirs(output_path, exist_ok=True)
12. df.to_csv(os.path.join(output_path, "{{table_name}}.csv"), index=False)
13. Print: f"Saved {{len(df)}} rows → {{path}}"

STYLE RULES (required for correctness):
- No inline comments or docstrings
- Never build columns inside pd.DataFrame({{...}}) with long inline lists
- Always: arr = np.random...; df['col'] = arr
- Use numpy vectorization — no Python for-loops for row generation

Output ONLY the Python source code.
{retry_block}"""

    response = _llm().invoke([SystemMessage(content=CODE_SYSTEM), HumanMessage(content=prompt)])
    code = response.content.strip()
    code = re.sub(r"^```(?:python)?\s*\n?", "", code, flags=re.IGNORECASE)
    code = re.sub(r"\n?```\s*$", "", code).strip()

    return {
        "messages": [AIMessage(content=f"[Attempt {attempt_num}] Code written ({len(code)} chars)")],
        "code": code,
        "attempts": attempt_num,
        "error": "",
    }


# ── Node 4: run_code ─────────────────────────────────────────────────────────

def run_code(state: State) -> dict:
    print("[Executing generated code...]")
    try:
        compiled = compile(state["code"], "<generated>", "exec")
    except SyntaxError as e:
        error_text = (
            f"SyntaxError (code was likely truncated): {e}\n\n"
            "Fix: write shorter, more vectorized code. "
            "Never use long pd.DataFrame({{...}}) inline constructors. "
            "Build each column as a numpy array then assign: df['col'] = arr."
        )
        return {"messages": [AIMessage(content=f"Syntax error: {e}")], "error": error_text}

    namespace: dict = {}
    try:
        exec(compiled, namespace)  # noqa: S102
        try:
            csv_files = [f for f in os.listdir(state["output_path"]) if f.endswith(".csv")]
        except FileNotFoundError:
            csv_files = []
        summary = f"Success — {len(csv_files)} CSV file(s) in '{state['output_path']}': {csv_files}"
        return {"messages": [AIMessage(content=summary)], "error": ""}
    except Exception:
        error_text = traceback.format_exc()
        return {"messages": [AIMessage(content=f"Execution error:\n{error_text}")], "error": error_text}


def should_retry(state: State) -> str:
    if not state.get("error"):
        return "success"
    if state.get("attempts", 0) < MAX_RETRIES:
        return "retry"
    return "give_up"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    memory = MemorySaver()
    builder = StateGraph(State)

    builder.add_node("gather_requirements", gather_requirements)
    builder.add_node("design_schema", design_schema)
    builder.add_node("write_code", write_code)
    builder.add_node("run_code", run_code)

    builder.add_edge(START, "gather_requirements")
    builder.add_conditional_edges(
        "gather_requirements",
        should_gather_more,
        {"gather": "gather_requirements", "design": "design_schema"},
    )
    builder.add_edge("design_schema", "write_code")
    builder.add_edge("write_code", "run_code")
    builder.add_conditional_edges(
        "run_code",
        should_retry,
        {"success": END, "retry": "write_code", "give_up": END},
    )

    return builder.compile(checkpointer=memory)
