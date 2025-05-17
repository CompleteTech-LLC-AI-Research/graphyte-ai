"""Generate the workflow graph description."""

from pathlib import Path


def write_workflow_graph() -> None:
    """Write the future workflow description to ``workflow.gv``.

    The script reads ``future_workflow.gv`` from the project root and writes the
    contents directly to ``workflow.gv``. This allows the graph to be rendered
    with Graphviz tools.
    """
    project_root = Path(__file__).resolve().parent.parent
    source_file = project_root / "future_workflow.gv"
    target_file = project_root / "workflow.gv"

    try:
        content = source_file.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - simple utility
        print(f"Error reading {source_file}: {exc}")
        return

    try:
        target_file.write_text(content, encoding="utf-8")
        print(f"Workflow graph written to {target_file}")
    except OSError as exc:  # pragma: no cover - simple utility
        print(f"Error writing {target_file}: {exc}")


if __name__ == "__main__":
    write_workflow_graph()
