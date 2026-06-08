import os
import re
import uuid
import time
import pandas as pd
import streamlit as st
from datetime import datetime
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from agent import build_graph
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── CSS ───────────────────────────────────────────────────────────────────────

def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        /* Chat messages */
        [data-testid="stChatMessage"] {
            border-radius: 10px;
            padding: 10px 16px;
            margin-bottom: 4px;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background-color: #161B22;
            border: 1px solid #30363D;
            border-radius: 8px;
            padding: 14px 18px;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #8B949E !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.4rem;
            font-weight: 700;
            color: #E6EDF3 !important;
        }

        /* Expanders */
        [data-testid="stExpander"] {
            border: 1px solid #21262D;
            border-radius: 8px;
        }
        [data-testid="stExpander"] summary {
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            color: #8B949E;
        }
        [data-testid="stExpander"] summary:hover {
            color: #E6EDF3;
        }

        /* Download buttons */
        [data-testid="stDownloadButton"] > button {
            width: 100%;
            background-color: #21262D;
            border: 1px solid #30363D;
            color: #E6EDF3;
            border-radius: 6px;
            font-size: 0.85rem;
            padding: 8px 16px;
            transition: border-color 0.15s ease;
        }
        [data-testid="stDownloadButton"] > button:hover {
            background-color: #30363D;
            border-color: #4F8EF7;
        }

        /* Tabs */
        [data-testid="stTabs"] [role="tab"] {
            font-size: 0.875rem;
            font-weight: 500;
            padding: 6px 14px;
        }

        /* Sidebar section labels */
        .sidebar-section {
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #8B949E;
            margin-top: 1.4rem;
            margin-bottom: 0.3rem;
        }

        /* Results divider */
        .results-header {
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #8B949E;
            border-bottom: 1px solid #21262D;
            padding-bottom: 6px;
            margin-top: 1.6rem;
            margin-bottom: 1rem;
        }

        /* Page title */
        h1 {
            font-size: 1.55rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
        }

        /* Dataframe caption */
        [data-testid="stCaptionContainer"] {
            color: #8B949E;
            font-size: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Session state ─────────────────────────────────────────────────────────────

def init_session_state() -> None:
    defaults = {
        "chat_history": [],
        "agent_app": None,
        "agent_config": None,
        "thread_id": str(uuid.uuid4()),
        "agent_phase": "idle",
        "pending_interrupt": False,
        "output_path": "",
        "final_state": None,
        "csv_previews": [],
        "preview_rows": 100,
        "schema_expanded": False,
        "code_expanded": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ── Agent helpers ─────────────────────────────────────────────────────────────

def get_or_create_agent():
    if st.session_state["agent_app"] is None:
        app = build_graph()
        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
        st.session_state["agent_app"] = app
        st.session_state["agent_config"] = config
    return st.session_state["agent_app"], st.session_state["agent_config"]


def start_new_session() -> None:
    preserved = {}
    for key in ("preview_rows", "schema_expanded", "code_expanded"):
        preserved[key] = st.session_state.get(key)

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    init_session_state()
    for key, val in preserved.items():
        st.session_state[key] = val

    st.rerun()


def _make_output_path(user_input: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", user_input[:40].lower()).strip("_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"output_{slug}_{ts}"


def invoke_initial(user_input: str) -> None:
    app, config = get_or_create_agent()
    output_path = _make_output_path(user_input)
    st.session_state["output_path"] = output_path

    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "schema": {},
        "code": "",
        "output_path": output_path,
        "error": "",
        "attempts": 0,
    }

    with st.spinner("Thinking..."):
        app.invoke(initial_state, config)

    _process_snapshot()


def invoke_resume(user_answer: str) -> None:
    app, config = get_or_create_agent()

    # Each resume either produces another question quickly or runs the full generation pipeline.
    # The spinner covers both cases — it disappears fast if another question arrives,
    # or stays up for ~30-60s during the design+write+execute phase.
    with st.spinner("Generating your dataset — this may take a minute..."):
        app.invoke(Command(resume=user_answer), config)

    _process_snapshot()


def _process_snapshot() -> None:
    app, config = get_or_create_agent()
    snapshot = app.get_state(config)

    interrupts = [i for task in snapshot.tasks for i in task.interrupts]

    if snapshot.next and interrupts:
        question = interrupts[0].value
        _stream_to_chat(question)
        st.session_state["agent_phase"] = "gathering"
        st.session_state["pending_interrupt"] = True
    else:
        # Graph reached END
        final = snapshot.values
        st.session_state["agent_phase"] = "done"
        st.session_state["pending_interrupt"] = False
        st.session_state["final_state"] = final

        if not final.get("error"):
            _ingest_results()
            table_count = len(st.session_state["csv_previews"])
            total_rows = sum(t["total_rows"] for t in st.session_state["csv_previews"])
            msg = (
                f"Done. Generated {table_count} table{'s' if table_count != 1 else ''} "
                f"with {total_rows:,} total rows. See the results below."
            )
        else:
            attempts = final.get("attempts", 0)
            msg = (
                f"Generation failed after {attempts} attempt{'s' if attempts != 1 else ''}. "
                "See the error details below."
            )

        _stream_to_chat(msg)


def _stream_to_chat(text: str) -> None:
    def char_gen(s: str):
        for ch in s:
            yield ch
            time.sleep(0.006)

    with st.chat_message("assistant"):
        st.write_stream(char_gen(text))

    st.session_state["chat_history"].append({"role": "assistant", "content": text})


def _ingest_results() -> None:
    path = st.session_state["output_path"]
    preview_rows = st.session_state.get("preview_rows", 100)

    if not os.path.isdir(path):
        return

    previews = []
    for fname in sorted(f for f in os.listdir(path) if f.endswith(".csv")):
        fpath = os.path.join(path, fname)
        try:
            df_preview = pd.read_csv(fpath, nrows=preview_rows)
            # Count total rows without loading full file
            total_rows = sum(1 for _ in open(fpath, encoding="utf-8")) - 1
            file_size = os.path.getsize(fpath)
            previews.append({
                "name": fname,
                "path": fpath,
                "df_preview": df_preview,
                "total_rows": max(total_rows, len(df_preview)),
                "file_size_bytes": file_size,
                "col_count": len(df_preview.columns),
            })
        except Exception:
            pass

    st.session_state["csv_previews"] = previews


# ── Input handler ─────────────────────────────────────────────────────────────

def handle_user_input(text: str) -> None:
    text = text.strip()
    if not text:
        return

    st.session_state["chat_history"].append({"role": "user", "content": text})

    with st.chat_message("user"):
        st.write(text)

    phase = st.session_state["agent_phase"]

    if phase == "idle":
        invoke_initial(text)
    elif phase == "gathering" and st.session_state["pending_interrupt"]:
        invoke_resume(text)
    elif phase == "done":
        st.info("Dataset generation is complete. Start a new session from the sidebar to generate another.")
        return

    st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        st.title("Synthetic Data Generator")

        st.markdown('<div class="sidebar-section">Session</div>', unsafe_allow_html=True)
        if st.button("New Session", type="secondary", use_container_width=True):
            start_new_session()

        st.markdown('<div class="sidebar-section">Status</div>', unsafe_allow_html=True)
        phase = st.session_state.get("agent_phase", "idle")
        phase_labels = {
            "idle": "Waiting",
            "gathering": "Gathering",
            "done": "Complete",
        }
        st.metric("Phase", phase_labels.get(phase, phase.title()))
        tid = st.session_state.get("thread_id", "")
        st.caption(f"Thread: `{tid[:8]}…`")

        st.markdown('<div class="sidebar-section">Display</div>', unsafe_allow_html=True)
        st.slider("Preview rows", 10, 200, 100, key="preview_rows")
        st.checkbox("Expand schema by default", key="schema_expanded")
        st.checkbox("Expand code by default", key="code_expanded")

        st.markdown('<div class="sidebar-section">About</div>', unsafe_allow_html=True)
        st.caption("Powered by Google Gemini 2.5 Flash")
        st.caption("LangGraph StateGraph + MemorySaver")
        st.caption("Built with Streamlit")


# ── Chat history ──────────────────────────────────────────────────────────────

def render_chat_history() -> None:
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


# ── Results panel ─────────────────────────────────────────────────────────────

def _format_size(bytes_: int) -> str:
    if bytes_ >= 1_000_000:
        return f"{bytes_ / 1_000_000:.1f} MB"
    if bytes_ >= 1_000:
        return f"{bytes_ / 1_000:.0f} KB"
    return f"{bytes_} B"


def render_metrics_row(csv_previews: list) -> None:
    final = st.session_state.get("final_state") or {}
    attempts = final.get("attempts", 1)
    total_rows = sum(t["total_rows"] for t in csv_previews)
    total_size = sum(t["file_size_bytes"] for t in csv_previews)

    cols = st.columns(4)
    cols[0].metric("Tables generated", len(csv_previews))
    cols[1].metric("Total rows", f"{total_rows:,}")
    cols[2].metric("Total size", _format_size(total_size))
    cols[3].metric("Attempts", attempts)

    if len(csv_previews) > 1:
        st.markdown("")
        per_cols = st.columns(len(csv_previews))
        for col, table in zip(per_cols, csv_previews):
            with col:
                st.metric(
                    table["name"].replace(".csv", ""),
                    f"{table['total_rows']:,} rows",
                    f"{table['col_count']} cols · {_format_size(table['file_size_bytes'])}",
                )


def render_schema_expander(schema: dict) -> None:
    if not schema:
        return
    with st.expander("Schema design", expanded=st.session_state.get("schema_expanded", False)):
        st.json(schema)


def render_code_expander(code: str) -> None:
    if not code:
        return
    with st.expander(
        f"Generated Python code  ({len(code):,} chars)",
        expanded=st.session_state.get("code_expanded", False),
    ):
        st.code(code, language="python", line_numbers=True)


def render_csv_tabs(csv_previews: list) -> None:
    tab_labels = [t["name"].replace(".csv", "") for t in csv_previews]
    tabs = st.tabs(tab_labels)

    preview_rows = st.session_state.get("preview_rows", 100)

    for tab, table in zip(tabs, csv_previews):
        with tab:
            shown = min(preview_rows, len(table["df_preview"]))
            st.caption(
                f"Showing {shown} of {table['total_rows']:,} rows  ·  "
                f"{table['col_count']} columns  ·  {_format_size(table['file_size_bytes'])}"
            )
            st.dataframe(
                table["df_preview"].head(preview_rows),
                use_container_width=True,
                height=360,
            )
            st.markdown("")
            try:
                with open(table["path"], "rb") as f:
                    data = f.read()
                st.download_button(
                    label=f"Download {table['name']}  ({_format_size(table['file_size_bytes'])})",
                    data=data,
                    file_name=table["name"],
                    mime="text/csv",
                    key=f"dl_{table['name']}",
                )
            except OSError:
                st.warning(f"File not found: {table['path']}")


def render_results_panel() -> None:
    if st.session_state["agent_phase"] != "done":
        return

    final = st.session_state.get("final_state") or {}

    if final.get("error"):
        st.error(f"Generation failed after {final.get('attempts', '?')} attempt(s).")
        with st.expander("Error details", expanded=True):
            st.code(final["error"], language="text")
        return

    csv_previews = st.session_state.get("csv_previews", [])
    if not csv_previews:
        return

    st.markdown('<div class="results-header">Results</div>', unsafe_allow_html=True)

    render_metrics_row(csv_previews)

    st.markdown("")

    render_schema_expander(final.get("schema", {}))
    render_code_expander(final.get("code", ""))

    st.markdown("")

    render_csv_tabs(csv_previews)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    inject_custom_css()
    init_session_state()
    render_sidebar()

    st.title("Synthetic Data Generator")
    st.caption(
        "Describe your domain and the agent will ask a few clarifying questions, "
        "then generate realistic, intentionally messy CSV datasets for ML/DL workflows."
    )

    st.divider()

    # Show a hint when the session is fresh
    if not st.session_state["chat_history"]:
        st.markdown(
            """
            <div style="color: #8B949E; font-size: 0.9rem; margin-bottom: 1rem;">
            Examples to get started:<br>
            &nbsp;&nbsp;· <em>E-commerce platform with customers, orders, and products</em><br>
            &nbsp;&nbsp;· <em>Credit card transactions for fraud detection</em><br>
            &nbsp;&nbsp;· <em>Hospital patient records and clinical notes</em>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_chat_history()
    render_results_panel()

    prompt = st.chat_input("Describe your domain and dataset needs…")
    if prompt:
        handle_user_input(prompt)


if __name__ == "__main__":
    main()
