"""Orchestrator module for the agentic team workflow.

This module coordinates the workflow between different agent steps and handles tracing.
"""

# ruff: noqa: E402


import asyncio  # Added for gather
import logging
import sys
from typing import Any, Optional, List


# Get logger for this module
logger = logging.getLogger(__name__)

# --- SDK Imports ---
try:
    from agents import gen_trace_id, trace  # type: ignore[attr-defined]
except ImportError:
    logger.critical(
        "CRITICAL Error: 'agents' SDK library not found or incomplete. Cannot define orchestrator.",
        exc_info=True,
    )
    print(
        "CRITICAL Error: 'agents' SDK library not found or incomplete. Cannot define orchestrator.",
        file=sys.stderr,
    )
    raise  # Re-raise the import error

# Import local configuration
from .config import (  # noqa: E402
    DOMAIN_MODEL,
    SUB_DOMAIN_MODEL,
    TOPIC_MODEL,
    ENTITY_TYPE_MODEL,
    EVENT_TYPE_MODEL,
    STATEMENT_TYPE_MODEL,
    EVIDENCE_TYPE_MODEL,
    MEASUREMENT_TYPE_MODEL,
    ENTITY_INSTANCE_MODEL,
    EVENT_INSTANCE_MODEL,
    STATEMENT_INSTANCE_MODEL,
    EVIDENCE_INSTANCE_MODEL,
    MEASUREMENT_INSTANCE_MODEL,
    RELATIONSHIP_MODEL,
    RELATIONSHIP_INSTANCE_MODEL,
    AGENT_TRACE_BASE_URL,
)
from .schemas import (  # noqa: E402
    DomainResultSchema,  # noqa: F401 - used in step1 output typing
    EntityTypeSchema,
    OntologyTypeSchema,
    EventTypeSchema,
    StatementTypeSchema,
    EvidenceTypeSchema,
    MeasurementTypeSchema,
    ModalityTypeSchema,
)


# Import steps
from .steps import (  # noqa: E402
    identify_domain,
    identify_subdomains,
    identify_topics,
    identify_entity_types,
    identify_ontology_types,
    identify_event_types,
    identify_statement_types,
    identify_evidence_types,
    identify_measurement_types,
    identify_modality_types,  # Added import for new step (4g)
    identify_entity_instances,
    identify_ontology_instances,
    identify_event_instances,
    identify_statement_instances,
    identify_evidence_instances,
    identify_measurement_instances,
    identify_modality_instances,
    aggregate_extracted_instances,
    identify_relationship_types,
    identify_relationship_instances,
)


# --- Helper to Run Steps with Individual Traces ---
from typing import Callable, Awaitable


async def run_step_with_trace(
    step_func: Callable[..., Awaitable[Any] | Any],
    step_name: str,
    overall_group_id: str,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, str]:
    """Run a workflow step wrapped in its own trace span.

    Args:
        step_func: The step function to execute.
        step_name: Name of the workflow step.
        overall_group_id: Group ID linking all step traces.
        *args: Positional arguments for the step.
        **kwargs: Keyword arguments for the step.

    Returns:
        Tuple containing the step's output and its trace ID.
    """

    step_trace_id = gen_trace_id()
    metadata = {"workflow_step": step_name}
    logger.info(f"Starting {step_name} (Trace ID: {step_trace_id})")
    with trace(
        workflow_name=step_name,
        group_id=overall_group_id,
        trace_id=step_trace_id,
        metadata=metadata,
    ):
        result_obj = step_func(
            *args, trace_id=step_trace_id, group_id=overall_group_id, **kwargs
        )
        if asyncio.iscoroutine(result_obj):
            result_val = await result_obj
        else:
            result_val = result_obj
    return result_val, step_trace_id


# --- Main Execution Logic (Combined Workflow in Single Trace) ---
async def run_combined_workflow(content: str) -> None:
    """Runs domain, sub-domain, topic, entity, ontology, event, statement, evidence,
    measurement, modality, entity, ontology, event, statement, evidence, measurement instance extraction, modality instance extraction, and relationship identification within a single trace.
    """
    # Skip processing if input content is empty or only whitespace
    if not content or not content.strip():
        logger.warning("Input content is empty or whitespace only. Skipping analysis.")
        print("Input content is empty. No analysis performed.")
        return

    # Initialize variables to store results from each step
    overall_trace_id: Optional[str] = None
    domain_data: Optional[DomainResultSchema] = None
    sub_domain_data = None
    topic_data = None
    entity_data = None
    ontology_data = None
    event_data = None
    statement_data = None
    evidence_data = None
    measurement_data = None
    modality_data = None  # Added variable for new step (4g)
    instance_data = None
    ontology_instance_data = None
    event_instance_data = None
    statement_instance_data = None
    evidence_instance_data = None
    measurement_instance_data = None
    modality_instance_data = None
    aggregated_instance_data = None
    relationship_data = None
    relationship_instance_data = None
    primary_domain = None

    # Generate a group ID to link all step traces
    overall_group_id = gen_trace_id()

    # Metadata for the single overall trace
    overall_trace_metadata = {
        "workflow_name": "Document Analysis",
        # "input_content_length": str(len(content)),
        # "start_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "domain_model": DOMAIN_MODEL,
        "sub_domain_model": SUB_DOMAIN_MODEL,
        "topic_model": TOPIC_MODEL,
        "entity_type_model": ENTITY_TYPE_MODEL,
        # "ontology_type_model": ONTOLOGY_TYPE_MODEL,
        "event_type_model": EVENT_TYPE_MODEL,
        "statement_type_model": STATEMENT_TYPE_MODEL,
        "evidence_type_model": EVIDENCE_TYPE_MODEL,
        "measurement_type_model": MEASUREMENT_TYPE_MODEL,
        # "modality_type_model": MODALITY_TYPE_MODEL,  # Added modality model (4g)
        "entity_instance_model": ENTITY_INSTANCE_MODEL,
        # "ontology_instance_model": ONTOLOGY_INSTANCE_MODEL,
        "event_instance_model": EVENT_INSTANCE_MODEL,
        "statement_instance_model": STATEMENT_INSTANCE_MODEL,
        "evidence_instance_model": EVIDENCE_INSTANCE_MODEL,
        "measurement_instance_model": MEASUREMENT_INSTANCE_MODEL,
        # "modality_instance_model": MODALITY_INSTANCE_MODEL,
        "relationship_model": RELATIONSHIP_MODEL,
        "relationship_instance_model": RELATIONSHIP_INSTANCE_MODEL,
    }

    # Start the overall trace for the entire workflow
    logger.info(
        f"--- Starting Analysis Workflow Trace ({overall_trace_metadata['workflow_name']}) ---"
    )
    print(f"\n--- Starting Workflow: {overall_trace_metadata['workflow_name']} ---")

    try:
        # Use the SDK's trace context manager to wrap the entire workflow
        with trace(
            overall_trace_metadata["workflow_name"],
            group_id=overall_group_id,
            metadata=overall_trace_metadata,
        ) as overall_span:
            # Attempt to get the trace ID and construct the URL
            if overall_span and hasattr(overall_span, "trace_id"):
                overall_trace_id = str(overall_span.trace_id)
                trace_url = f"{AGENT_TRACE_BASE_URL.rstrip('/')}/{overall_trace_id}"
                logger.info(f"Overall Workflow Trace URL: {trace_url}")
                print(f"Overall Workflow Trace URL: {trace_url}")
            else:
                logger.warning(
                    "Could not obtain overall workflow trace ID from context span."
                )
                print("Overall Workflow Trace ID not available.")

            # === Step 1: Identify Primary Domain (with Confidence) ===
            domain_result = await run_step_with_trace(
                identify_domain,
                "step1_domain",
                overall_group_id,
                content,
            )
            domain_data, step1_trace_id = domain_result

            primary_domain = domain_data.domain.strip() if domain_data else None

            # === Step 2: Identify Sub-Domains (with Relevance) ===
            sub_domain_result = (
                await run_step_with_trace(
                    identify_subdomains,
                    "step2_subdomains",
                    overall_group_id,
                    content,
                    primary_domain,
                )
                if primary_domain
                else None
            )
            sub_domain_data, step2_trace_id = (
                sub_domain_result if sub_domain_result else (None, "")
            )

            # === Step 3: Identify Topics in PARALLEL for each Sub-Domain (with Relevance) ===
            topic_result = (
                await run_step_with_trace(
                    identify_topics,
                    "step3_topics",
                    overall_group_id,
                    content,
                    primary_domain,
                    sub_domain_data,
                )
                if primary_domain and sub_domain_data
                else None
            )
            topic_data, step3_trace_id = topic_result if topic_result else (None, "")

            # === Step 4: Parallel Identification (Entities, Ontology, Events, Statements, Evidence, Measurements, Modalities) ===
            if primary_domain and sub_domain_data and topic_data:
                logger.info(
                    "--- Starting Step 4: Parallel Identification (Entities, Ontology, Events, Statements, Evidence, Measurements, Modalities) ---"
                )
                print("\n--- Starting Step 4: Parallel Identification ---")
                step4_tasks = [
                    run_step_with_trace(
                        identify_entity_types,
                        "step4a_entity_types",
                        overall_group_id,
                        content,
                        primary_domain,
                        sub_domain_data,
                        topic_data,
                    ),
                    run_step_with_trace(
                        identify_ontology_types,
                        "step4b_ontology_types",
                        overall_group_id,
                        content,
                        primary_domain,
                        sub_domain_data,
                        topic_data,
                    ),
                    run_step_with_trace(
                        identify_event_types,
                        "step4c_event_types",
                        overall_group_id,
                        content,
                        primary_domain,
                        sub_domain_data,
                        topic_data,
                    ),
                    run_step_with_trace(
                        identify_statement_types,
                        "step4d_statement_types",
                        overall_group_id,
                        content,
                        primary_domain,
                        sub_domain_data,
                        topic_data,
                    ),
                    run_step_with_trace(
                        identify_evidence_types,
                        "step4e_evidence_types",
                        overall_group_id,
                        content,
                        primary_domain,
                        sub_domain_data,
                        topic_data,
                    ),
                    run_step_with_trace(
                        identify_measurement_types,
                        "step4f_measurement_types",
                        overall_group_id,
                        content,
                        primary_domain,
                        sub_domain_data,
                        topic_data,
                    ),
                    run_step_with_trace(
                        identify_modality_types,
                        "step4g_modality_types",
                        overall_group_id,
                        content,
                        primary_domain,
                        sub_domain_data,
                        topic_data,
                    ),  # New task (4g)
                ]
                # The return type annotation is tricky here because gather returns a list of results OR exceptions
                # Using List[Any] is simpler for now, specific handling follows
                step4_results: List[Any] = await asyncio.gather(
                    *step4_tasks, return_exceptions=True
                )

                # Process results safely
                potential_entity_result = step4_results[0]
                potential_ontology_result = step4_results[1]
                potential_event_result = step4_results[2]
                potential_statement_result = step4_results[3]
                potential_evidence_result = step4_results[4]
                potential_measurement_result = step4_results[5]
                potential_modality_result = step4_results[6]  # New result (4g)

                if isinstance(potential_entity_result, Exception):
                    potential_entity_data = potential_entity_result
                    step4a_trace_id = ""
                else:
                    potential_entity_data, step4a_trace_id = potential_entity_result

                if isinstance(potential_ontology_result, Exception):
                    potential_ontology_data = potential_ontology_result
                    step4b_trace_id = ""
                else:
                    potential_ontology_data, step4b_trace_id = potential_ontology_result

                if isinstance(potential_event_result, Exception):
                    potential_event_data = potential_event_result
                    step4c_trace_id = ""
                else:
                    potential_event_data, step4c_trace_id = potential_event_result

                if isinstance(potential_statement_result, Exception):
                    potential_statement_data = potential_statement_result
                    step4d_trace_id = ""
                else:
                    potential_statement_data, step4d_trace_id = (
                        potential_statement_result
                    )

                if isinstance(potential_evidence_result, Exception):
                    potential_evidence_data = potential_evidence_result
                    step4e_trace_id = ""
                else:
                    potential_evidence_data, step4e_trace_id = potential_evidence_result

                if isinstance(potential_measurement_result, Exception):
                    potential_measurement_data = potential_measurement_result
                    step4f_trace_id = ""
                else:
                    potential_measurement_data, step4f_trace_id = (
                        potential_measurement_result
                    )

                if isinstance(potential_modality_result, Exception):
                    potential_modality_data = potential_modality_result
                    step4g_trace_id = ""
                else:
                    potential_modality_data, step4g_trace_id = potential_modality_result

                # Entity Data Processing
                if isinstance(potential_entity_data, Exception):
                    logger.error(
                        f"Step 4a (Entity Types) failed with exception: {potential_entity_data}",
                        exc_info=potential_entity_data,
                    )
                    print(
                        f"Error in Step 4a (Entity Types): {type(potential_entity_data).__name__}: {potential_entity_data}"
                    )
                    entity_data = None
                elif (
                    isinstance(potential_entity_data, EntityTypeSchema)
                    or potential_entity_data is None
                ):
                    entity_data = potential_entity_data
                else:
                    logger.error(
                        f"Step 4a (Entity Types) returned unexpected type: {type(potential_entity_data)}"
                    )
                    entity_data = None

                # Ontology Data Processing
                if isinstance(potential_ontology_data, Exception):
                    logger.error(
                        f"Step 4b (Ontology Types) failed with exception: {potential_ontology_data}",
                        exc_info=potential_ontology_data,
                    )
                    print(
                        f"Error in Step 4b (Ontology Types): {type(potential_ontology_data).__name__}: {potential_ontology_data}"
                    )
                    ontology_data = None
                elif (
                    isinstance(potential_ontology_data, OntologyTypeSchema)
                    or potential_ontology_data is None
                ):
                    ontology_data = potential_ontology_data
                else:
                    logger.error(
                        f"Step 4b (Ontology Types) returned unexpected type: {type(potential_ontology_data)}"
                    )
                    ontology_data = None

                # Event Data Processing
                if isinstance(potential_event_data, Exception):
                    logger.error(
                        f"Step 4c (Event Types) failed with exception: {potential_event_data}",
                        exc_info=potential_event_data,
                    )
                    print(
                        f"Error in Step 4c (Event Types): {type(potential_event_data).__name__}: {potential_event_data}"
                    )
                    event_data = None
                elif (
                    isinstance(potential_event_data, EventTypeSchema)
                    or potential_event_data is None
                ):
                    event_data = potential_event_data
                else:
                    logger.error(
                        f"Step 4c (Event Types) returned unexpected type: {type(potential_event_data)}"
                    )
                    event_data = None

                # Statement Data Processing
                if isinstance(potential_statement_data, Exception):
                    logger.error(
                        f"Step 4d (Statement Types) failed with exception: {potential_statement_data}",
                        exc_info=potential_statement_data,
                    )
                    print(
                        f"Error in Step 4d (Statement Types): {type(potential_statement_data).__name__}: {potential_statement_data}"
                    )
                    statement_data = None
                elif (
                    isinstance(potential_statement_data, StatementTypeSchema)
                    or potential_statement_data is None
                ):
                    statement_data = potential_statement_data
                else:
                    logger.error(
                        f"Step 4d (Statement Types) returned unexpected type: {type(potential_statement_data)}"
                    )
                    statement_data = None

                # Evidence Data Processing
                if isinstance(potential_evidence_data, Exception):
                    logger.error(
                        f"Step 4e (Evidence Types) failed with exception: {potential_evidence_data}",
                        exc_info=potential_evidence_data,
                    )
                    print(
                        f"Error in Step 4e (Evidence Types): {type(potential_evidence_data).__name__}: {potential_evidence_data}"
                    )
                    evidence_data = None
                elif (
                    isinstance(potential_evidence_data, EvidenceTypeSchema)
                    or potential_evidence_data is None
                ):
                    evidence_data = potential_evidence_data
                else:
                    logger.error(
                        f"Step 4e (Evidence Types) returned unexpected type: {type(potential_evidence_data)}"
                    )
                    evidence_data = None

                # Measurement Data Processing
                if isinstance(potential_measurement_data, Exception):
                    logger.error(
                        f"Step 4f (Measurement Types) failed with exception: {potential_measurement_data}",
                        exc_info=potential_measurement_data,
                    )
                    print(
                        f"Error in Step 4f (Measurement Types): {type(potential_measurement_data).__name__}: {potential_measurement_data}"
                    )
                    measurement_data = None
                elif (
                    isinstance(potential_measurement_data, MeasurementTypeSchema)
                    or potential_measurement_data is None
                ):
                    measurement_data = potential_measurement_data
                else:
                    logger.error(
                        f"Step 4f (Measurement Types) returned unexpected type: {type(potential_measurement_data)}"
                    )
                    measurement_data = None

                # Modality Data Processing (New Step 4g)
                if isinstance(potential_modality_data, Exception):
                    logger.error(
                        f"Step 4g (Modality Types) failed with exception: {potential_modality_data}",
                        exc_info=potential_modality_data,
                    )
                    print(
                        f"Error in Step 4g (Modality Types): {type(potential_modality_data).__name__}: {potential_modality_data}"
                    )
                    modality_data = None
                elif (
                    isinstance(potential_modality_data, ModalityTypeSchema)
                    or potential_modality_data is None
                ):
                    modality_data = potential_modality_data
                else:
                    logger.error(
                        f"Step 4g (Modality Types) returned unexpected type: {type(potential_modality_data)}"
                    )
                    modality_data = None

                logger.info("--- Finished Step 4: Parallel Identification ---")
                print("--- Finished Step 4: Parallel Identification ---")
            else:
                logger.warning(
                    "Skipping Step 4 (Parallel ID) because prerequisites were not met."
                )
                print("Skipping Step 4 (Parallel ID) due to missing prior results.")

            # === Step 5a: Extract Specific Entity Instances ===
            instance_result = (
                await run_step_with_trace(
                    identify_entity_instances,
                    "step5a_entity_instances",
                    overall_group_id,
                    content,
                    primary_domain,
                    sub_domain_data,
                    topic_data,
                    entity_data,
                )
                if primary_domain and sub_domain_data and topic_data and entity_data
                else None
            )
            instance_data, step5a_trace_id = (
                instance_result if instance_result else (None, "")
            )

            # === Step 5b: Extract Ontology Concept Instances ===
            ontology_instance_result = (
                await run_step_with_trace(
                    identify_ontology_instances,
                    "step5b_ontology_instances",
                    overall_group_id,
                    content,
                    primary_domain,
                    sub_domain_data,
                    topic_data,
                    ontology_data,
                )
                if primary_domain and sub_domain_data and topic_data and ontology_data
                else None
            )
            ontology_instance_data, step5b_trace_id = (
                ontology_instance_result if ontology_instance_result else (None, "")
            )

            # === Step 5c: Extract Event Instances ===
            event_instance_result = (
                await run_step_with_trace(
                    identify_event_instances,
                    "step5c_event_instances",
                    overall_group_id,
                    content,
                    primary_domain,
                    sub_domain_data,
                    topic_data,
                    event_data,
                )
                if primary_domain and sub_domain_data and topic_data and event_data
                else None
            )
            event_instance_data, step5c_trace_id = (
                event_instance_result if event_instance_result else (None, "")
            )

            # === Step 5d: Extract Statement Instances ===
            statement_instance_result = (
                await run_step_with_trace(
                    identify_statement_instances,
                    "step5d_statement_instances",
                    overall_group_id,
                    content,
                    primary_domain,
                    sub_domain_data,
                    topic_data,
                    statement_data,
                )
                if primary_domain and sub_domain_data and topic_data and statement_data
                else None
            )
            statement_instance_data, step5d_trace_id = (
                statement_instance_result if statement_instance_result else (None, "")
            )

            # === Step 5e: Extract Evidence Instances ===
            evidence_instance_result = (
                await run_step_with_trace(
                    identify_evidence_instances,
                    "step5e_evidence_instances",
                    overall_group_id,
                    content,
                    primary_domain,
                    sub_domain_data,
                    topic_data,
                    evidence_data,
                )
                if primary_domain and sub_domain_data and topic_data and evidence_data
                else None
            )
            evidence_instance_data, step5e_trace_id = (
                evidence_instance_result if evidence_instance_result else (None, "")
            )

            # === Step 5f: Extract Measurement Instances ===
            measurement_instance_result = (
                await run_step_with_trace(
                    identify_measurement_instances,
                    "step5f_measurement_instances",
                    overall_group_id,
                    content,
                    primary_domain,
                    sub_domain_data,
                    topic_data,
                    measurement_data,
                )
                if primary_domain
                and sub_domain_data
                and topic_data
                and measurement_data
                else None
            )
            measurement_instance_data, step5f_trace_id = (
                measurement_instance_result
                if measurement_instance_result
                else (None, "")
            )

            # === Step 5g: Extract Modality Instances ===
            modality_instance_result = (
                await run_step_with_trace(
                    identify_modality_instances,
                    "step5g_modality_instances",
                    overall_group_id,
                    content,
                    primary_domain,
                    sub_domain_data,
                    topic_data,
                    modality_data,
                )
                if primary_domain and sub_domain_data and topic_data and modality_data
                else None
            )
            modality_instance_data, step5g_trace_id = (
                modality_instance_result if modality_instance_result else (None, "")
            )

            # === Step 6: Identify Relationships in PARALLEL for each Entity Type (Based on Context) ===
            # Note: This step currently only uses entity_data. If relationships involving other types were needed,
            # the step would require modification to accept and use that data.
            relationship_result = (
                await run_step_with_trace(
                    identify_relationship_types,
                    "step6a_relationship_types",
                    overall_group_id,
                    content,
                    primary_domain,
                    sub_domain_data,
                    topic_data,
                    entity_data,
                )
                if primary_domain and sub_domain_data and topic_data and entity_data
                else None
            )
            relationship_data, step6a_trace_id = (
                relationship_result if relationship_result else (None, "")
            )

            relationship_instance_result = (
                await run_step_with_trace(
                    identify_relationship_instances,
                    "step6b_relationship_instances",
                    overall_group_id,
                    content,
                    primary_domain,
                    sub_domain_data,
                    relationship_data,
                )
                if primary_domain and sub_domain_data and relationship_data
                else None
            )
            relationship_instance_data, step6b_trace_id = (
                relationship_instance_result
                if relationship_instance_result
                else (None, "")
            )

            # === Aggregate Extracted Instances (Steps 5a-5g + Relationships) ===
            agg_result, step6c_trace_id = await run_step_with_trace(
                aggregate_extracted_instances,
                "step6c_aggregate_instances",
                overall_group_id,
                primary_domain,
                sub_domain_data,
                instance_data,
                ontology_instance_data,
                event_instance_data,
                statement_instance_data,
                evidence_instance_data,
                measurement_instance_data,
                modality_instance_data,
                relationship_instance_data,
            )
            aggregated_instance_data = agg_result

            # Log completion status of individual steps (optional)
            logger.info(
                f"Step 1 (Domain) Result: {'Success' if domain_data else 'Failed/Skipped'}"
            )
            logger.info(
                f"Step 2 (SubDomain) Result: {'Success' if sub_domain_data else 'Failed/Skipped'}"
            )
            logger.info(
                f"Step 3 (Topics) Result: {'Success' if topic_data else 'Failed/Skipped'}"
            )
            logger.info(
                f"Step 4a (Entity Types) Result: {'Success' if entity_data else 'Failed/Skipped/Error'}"
            )
            logger.info(
                f"Step 4b (Ontology Types) Result: {'Success' if ontology_data else 'Failed/Skipped/Error'}"
            )
            logger.info(
                f"Step 4c (Event Types) Result: {'Success' if event_data else 'Failed/Skipped/Error'}"
            )
            logger.info(
                f"Step 4d (Statement Types) Result: {'Success' if statement_data else 'Failed/Skipped/Error'}"
            )
            logger.info(
                f"Step 4e (Evidence Types) Result: {'Success' if evidence_data else 'Failed/Skipped/Error'}"
            )
            logger.info(
                f"Step 4f (Measurement Types) Result: {'Success' if measurement_data else 'Failed/Skipped/Error'}"
            )
            logger.info(
                f"Step 4g (Modality Types) Result: {'Success' if modality_data else 'Failed/Skipped/Error'}"
            )  # Added log for new step (4g)
            logger.info(
                f"Step 5a (Entity Instances) Result: {'Success' if instance_data else 'Failed/Skipped'}"
            )
            logger.info(
                f"Step 5b (Ontology Instances) Result: {'Success' if ontology_instance_data else 'Failed/Skipped'}"
            )
            logger.info(
                f"Step 5c (Event Instances) Result: {'Success' if event_instance_data else 'Failed/Skipped'}"
            )
            logger.info(
                f"Step 5d (Statement Instances) Result: {'Success' if statement_instance_data else 'Failed/Skipped'}"
            )
            logger.info(
                f"Step 5e (Evidence Instances) Result: {'Success' if evidence_instance_data else 'Failed/Skipped'}"
            )
            logger.info(
                f"Step 5f (Measurement Instances) Result: {'Success' if measurement_instance_data else 'Failed/Skipped'}"
            )
            logger.info(
                f"Step 5g (Modality Instances) Result: {'Success' if modality_instance_data else 'Failed/Skipped'}"
            )
            logger.info(
                f"Step 6 (Relationships) Result: {'Success' if relationship_data else 'Failed/Skipped'}"
            )
            logger.info(
                f"Step 6b (Relationship Instances) Result: {'Success' if relationship_instance_data else 'Failed/Skipped'}"
            )
            logger.info(
                f"Aggregated Instances Result: {'Success' if aggregated_instance_data else 'Failed/Skipped'}"
            )

    except Exception as e:
        # Catch errors occurring outside the specific steps but within the trace
        logger.exception(
            "An unexpected error occurred within the main workflow trace.",
            extra={"trace_id": overall_trace_id or "N/A"},
        )
        print(
            f"\nAn unexpected error occurred during the main workflow: {type(e).__name__}: {e}"
        )

    # This message prints regardless of success/failure within the trace
    print(f"\nFull Workflow ({overall_trace_metadata['workflow_name']}) finished.")
    logger.info(
        f"--- Finished Analysis Workflow Trace ({overall_trace_metadata['workflow_name']}) (ID: {overall_trace_id or 'N/A'}) ---"
    )
