import os
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from agent import build_graph

load_dotenv()


def main():
    print("Synthetic Data Generator — LangGraph + Gemini 2.5 Flash")
    print("=" * 55)
    domain = input("Describe your domain and dataset needs:\n> ").strip()
    if not domain:
        domain = "e-commerce platform with customers, orders, and products"

    slug = re.sub(r"[^a-z0-9]+", "_", domain.lower())[:40].strip("_")
    output_path = f"output_{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    app = build_graph()
    config = {"configurable": {"thread_id": "1"}}

    # Initial invoke — runs until the first interrupt (first clarifying question)
    app.invoke(
        {
            "messages": [HumanMessage(content=domain)],
            "schema": {},
            "code": "",
            "output_path": output_path,
            "error": "",
            "attempts": 0,
        },
        config=config,
    )

    # Drive the interrupt loop: answer questions until the graph finishes
    while True:
        snapshot = app.get_state(config)

        if not snapshot.next:
            # Graph has reached END
            break

        interrupts = [i for task in snapshot.tasks for i in task.interrupts]
        if not interrupts:
            # Paused at a node boundary with no interrupt — shouldn't happen, but safe to break
            break

        question = interrupts[0].value
        print(f"\n{question}")
        user_input = input("> ").strip()

        app.invoke(Command(resume=user_input), config=config)

    # ── Results ───────────────────────────────────────────────────────────────
    final = app.get_state(config).values
    print("\n" + "=" * 55)

    if final.get("error"):
        print("FAILED after max retries.")
        print(f"Last error:\n{final['error'][:600]}")
        print(f"Attempts: {final.get('attempts', 0)}")
    else:
        try:
            csv_files = sorted(f for f in os.listdir(output_path) if f.endswith(".csv"))
        except FileNotFoundError:
            csv_files = []

        print(f"SUCCESS — {len(csv_files)} CSV file(s) in '{output_path}':")
        for f in csv_files:
            path = os.path.join(output_path, f)
            size_kb = os.path.getsize(path) / 1024
            print(f"  {f}  ({size_kb:,.1f} KB)")
        print(f"Attempts: {final.get('attempts', 0)}")


if __name__ == "__main__":
    main()
