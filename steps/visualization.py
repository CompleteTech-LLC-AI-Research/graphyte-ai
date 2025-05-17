"""Visualization functionality for agent workflow."""

import logging
import sys

from ..agents import domain_identifier_agent
from ..config import VISUALIZATION_OUTPUT_DIR, VISUALIZATION_FILENAME

logger = logging.getLogger(__name__)

# Flag to track visualization extension availability
VIZ_AVAILABLE = False

# Attempt to import visualization extension
try:
    from agents.extensions.visualization import draw_graph

    VIZ_AVAILABLE = True
    logger.info(
        "Agent visualization extension ('agents.extensions.visualization') loaded successfully."
    )
except ImportError:
    # Define a dummy draw_graph if not available to avoid NameError later
    def draw_graph(*args, **kwargs):
        logger.error(
            "Visualization requested, but 'agentic_team[viz]' extras not installed, Graphviz is missing, or import failed (expected 'agents.extensions.visualization')."
        )
        print(
            "ERROR: Visualization generation failed. Please install 'agentic_team[viz]' (or similar extras) and ensure Graphviz is installed and in PATH.",
            file=sys.stderr,
        )
        return None

    logger.warning(
        "Could not import 'draw_graph' from 'agents.extensions.visualization'. Visualization will be unavailable. Ensure 'agentic_team[viz]' (or similar) is installed."
    )


async def generate_workflow_visualization() -> None:
    """Generates a visualization of the defined agent structure and saves it."""
    if not VIZ_AVAILABLE:
        # Error message already printed by dummy draw_graph or import failure logged
        print("Skipping visualization generation due to missing dependencies.")
        return  # Exit the function gracefully

    logger.info("--- Generating Agent Workflow Visualization ---")
    print("\n--- Generating Agent Workflow Visualization ---")

    # Ensure the output directory exists
    try:
        VISUALIZATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"Ensured visualization output directory exists: {VISUALIZATION_OUTPUT_DIR}"
        )
    except OSError as e:
        logger.exception(
            f"Failed to create visualization output directory {VISUALIZATION_OUTPUT_DIR}: {e}"
        )
        print(
            f"Error: Could not create output directory: {VISUALIZATION_OUTPUT_DIR}",
            file=sys.stderr,
        )
        return

    output_path = VISUALIZATION_OUTPUT_DIR / VISUALIZATION_FILENAME

    try:
        # --- Choose the Agent to Visualize ---
        # Option 1: Visualize the first agent. Simple, but shows limited structure.
        agent_to_visualize = domain_identifier_agent

        # Option 2 (More Complex): Create a dummy "Orchestrator" Agent *just for visualization*
        # This would involve defining an Agent with handoffs that *mimic* the flow in run_combined_workflow.
        # Example (Conceptual - requires defining tools/handoffs appropriately):
        # viz_orchestrator = Agent(
        #     name="WorkflowOrchestrator",
        #     instructions="Visualizes the workflow.",
        #     tools=[domain_identifier_agent.tools], # Example: aggregate tools if needed
        #     handoffs=[domain_identifier_agent, sub_domain_identifier_agent, ...] # Example: list agents
        # )
        # agent_to_visualize = viz_orchestrator
        # *Current Choice:* Use Option 1 for simplicity based on current structure.

        logger.info(f"Visualizing agent: {agent_to_visualize.name}")
        logger.info(f"Saving visualization graph to: {output_path}")
        print(f"Attempting to save graph to: {output_path}")

        # Generate and save the graph
        graph = draw_graph(
            agent_to_visualize, filename=str(output_path)
        )  # Pass filename directly

        if graph:  # draw_graph might return the graph object or None on failure
            # The function saves the file directly when filename is provided.
            # We can optionally view it if needed: graph.view()
            logger.info(
                f"Successfully generated and saved visualization graph to {output_path}"
            )
            print(f"Success: Visualization saved to {output_path}")
            print(
                "Note: The visualization shows static agent definitions (tools/handoffs). Complex orchestration logic (loops, parallel calls) within the code might not be fully represented."
            )
        else:
            # Error should have been logged by draw_graph (dummy or real)
            logger.error(
                f"draw_graph function failed to generate visualization for {agent_to_visualize.name}."
            )
            print(
                f"Error: Failed to generate visualization for {agent_to_visualize.name}. Check logs and Graphviz installation.",
                file=sys.stderr,
            )

    except Exception as e:
        logger.exception(
            f"An unexpected error occurred during visualization generation: {e}"
        )
        print(
            f"An unexpected error occurred during visualization: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
