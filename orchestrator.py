# File: /Users/completetech/Desktop/python-agent-sdk/src/agentic_team_workflow/orchestrator.py
"""Orchestrator module for the agentic team workflow.

This module coordinates the workflow between different agent steps and handles tracing.
"""

import asyncio # Added for gather
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple

from pydantic import ValidationError

# Get logger for this module
logger = logging.getLogger(__name__)

# --- SDK Imports ---
try:
    from agentic_team import trace
except ImportError:
    logger.critical("CRITICAL Error: 'agentic_team' SDK library not found or incomplete. Cannot define orchestrator.", exc_info=True)
    print("CRITICAL Error: 'agentic_team' SDK library not found or incomplete. Cannot define orchestrator.", file=sys.stderr)
    raise # Re-raise the import error

# Import local configuration
from .config import (
    DOMAIN_MODEL, SUB_DOMAIN_MODEL, TOPIC_MODEL,
    ENTITY_TYPE_MODEL, ONTOLOGY_TYPE_MODEL, EVENT_TYPE_MODEL, STATEMENT_TYPE_MODEL, EVIDENCE_TYPE_MODEL, MEASUREMENT_TYPE_MODEL, MODALITY_TYPE_MODEL, ENTITY_INSTANCE_MODEL, ONTOLOGY_INSTANCE_MODEL, EVENT_INSTANCE_MODEL, STATEMENT_INSTANCE_MODEL, EVIDENCE_INSTANCE_MODEL, MEASUREMENT_INSTANCE_MODEL, MODALITY_INSTANCE_MODEL, RELATIONSHIP_MODEL,
    AGENT_TRACE_BASE_URL
)
from .schemas import (
    EntityTypeSchema, OntologyTypeSchema, EventSchema, StatementTypeSchema, EvidenceTypeSchema,
    MeasurementTypeSchema, ModalityTypeSchema, EntityInstanceSchema, OntologyInstanceSchema, EventInstanceSchema, StatementInstanceSchema, EvidenceInstanceSchema, MeasurementInstanceSchema, ModalityInstanceSchema, RelationshipSchema
)

# Import steps
from .steps import (
    identify_domain,
    identify_subdomains,
    identify_topics,
    identify_entity_types,
    identify_ontology_types,
    identify_event_types,
    identify_statement_types,
    identify_evidence_types,
    identify_measurement_types,
    identify_modality_types, # Added import for new step (4g)
    identify_entity_instances,
    identify_ontology_instances,
    identify_event_instances,
    identify_statement_instances,
    identify_evidence_instances,
    identify_measurement_instances,
    identify_modality_instances,
    identify_relationship_types,
    generate_workflow_visualization
)

# --- Main Execution Logic (Combined Workflow in Single Trace) ---
async def run_combined_workflow(content: str) -> None:
    """Runs domain, sub-domain, topic, entity, ontology, event, statement, evidence,
    measurement, modality, entity, ontology, event, statement, evidence, measurement instance extraction, modality instance extraction, and relationship identification within a single trace."""
    # Skip processing if input content is empty or only whitespace
    if not content or not content.strip():
        logger.warning("Input content is empty or whitespace only. Skipping analysis.")
        print("Input content is empty. No analysis performed.")
        return

    # Initialize variables to store results from each step
    overall_trace_id: Optional[str] = None
    domain_data = None
    sub_domain_data = None
    topic_data = None
    entity_data = None
    ontology_data = None
    event_data = None
    statement_data = None
    evidence_data = None
    measurement_data = None
    modality_data = None # Added variable for new step (4g)
    instance_data = None
    ontology_instance_data = None
    event_instance_data = None
    statement_instance_data = None
    evidence_instance_data = None
    measurement_instance_data = None
    modality_instance_data = None
    relationship_data = None
    primary_domain = None

    # Metadata for the single overall trace
    overall_trace_metadata = {
        "workflow_name": "Document Analysis",
        "input_content_length": str(len(content)),
        "start_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "domain_model": DOMAIN_MODEL,
        "sub_domain_model": SUB_DOMAIN_MODEL,
        "topic_model": TOPIC_MODEL,
        "entity_type_model": ENTITY_TYPE_MODEL,
        "ontology_type_model": ONTOLOGY_TYPE_MODEL,
        "event_type_model": EVENT_TYPE_MODEL,
        "statement_type_model": STATEMENT_TYPE_MODEL,
        "evidence_type_model": EVIDENCE_TYPE_MODEL,
        "measurement_type_model": MEASUREMENT_TYPE_MODEL,
        "modality_type_model": MODALITY_TYPE_MODEL, # Added modality model (4g)
        "entity_instance_model": ENTITY_INSTANCE_MODEL,
        "ontology_instance_model": ONTOLOGY_INSTANCE_MODEL,
        "event_instance_model": EVENT_INSTANCE_MODEL,
        "statement_instance_model": STATEMENT_INSTANCE_MODEL,
        "evidence_instance_model": EVIDENCE_INSTANCE_MODEL,
        "measurement_instance_model": MEASUREMENT_INSTANCE_MODEL,
        "modality_instance_model": MODALITY_INSTANCE_MODEL,
        "relationship_model": RELATIONSHIP_MODEL,
    }

    # Start the overall trace for the entire workflow
    logger.info(f"--- Starting Analysis Workflow Trace ({overall_trace_metadata['workflow_name']}) ---")
    print(f"\n--- Starting Workflow: {overall_trace_metadata['workflow_name']} ---")

    try:
        # Use the SDK's trace context manager to wrap the entire workflow
        with trace(overall_trace_metadata["workflow_name"], metadata=overall_trace_metadata) as overall_span:
            # Attempt to get the trace ID and construct the URL
            if overall_span and hasattr(overall_span, 'trace_id'):
                overall_trace_id = str(overall_span.trace_id)
                trace_url = f"{AGENT_TRACE_BASE_URL.rstrip('/')}/{overall_trace_id}"
                logger.info(f"Overall Workflow Trace URL: {trace_url}")
                print(f"Overall Workflow Trace URL: {trace_url}")
            else:
                logger.warning("Could not obtain overall workflow trace ID from context span.")
                print("Overall Workflow Trace ID not available.")

            # === Step 1: Identify Primary Domain (with Confidence) ===
            domain_data = await identify_domain(content, overall_trace_id)
            primary_domain = domain_data.domain.strip() if domain_data else None

            # === Step 2: Identify Sub-Domains (with Relevance) ===
            sub_domain_data = await identify_subdomains(content, primary_domain, overall_trace_id) if primary_domain else None

            # === Step 3: Identify Topics in PARALLEL for each Sub-Domain (with Relevance) ===
            topic_data = await identify_topics(content, primary_domain, sub_domain_data, overall_trace_id) if primary_domain and sub_domain_data else None

            # === Step 4: Parallel Identification (Entities, Ontology, Events, Statements, Evidence, Measurements, Modalities) ===
            if primary_domain and sub_domain_data and topic_data:
                logger.info("--- Starting Step 4: Parallel Identification (Entities, Ontology, Events, Statements, Evidence, Measurements, Modalities) ---")
                print("\n--- Starting Step 4: Parallel Identification ---")
                step4_tasks = [
                    identify_entity_types(content, primary_domain, sub_domain_data, topic_data, overall_trace_id),
                    identify_ontology_types(content, primary_domain, sub_domain_data, topic_data, overall_trace_id),
                    identify_event_types(content, primary_domain, sub_domain_data, topic_data, overall_trace_id),
                    identify_statement_types(content, primary_domain, sub_domain_data, topic_data, overall_trace_id),
                    identify_evidence_types(content, primary_domain, sub_domain_data, topic_data, overall_trace_id),
                    identify_measurement_types(content, primary_domain, sub_domain_data, topic_data, overall_trace_id),
                    identify_modality_types(content, primary_domain, sub_domain_data, topic_data, overall_trace_id) # New task (4g)
                ]
                # The return type annotation is tricky here because gather returns a list of results OR exceptions
                # Using List[Any] is simpler for now, specific handling follows
                step4_results: List[Any] = await asyncio.gather(*step4_tasks, return_exceptions=True)

                # Process results safely
                potential_entity_data = step4_results[0]
                potential_ontology_data = step4_results[1]
                potential_event_data = step4_results[2]
                potential_statement_data = step4_results[3]
                potential_evidence_data = step4_results[4]
                potential_measurement_data = step4_results[5]
                potential_modality_data = step4_results[6] # New result (4g)

                # Entity Data Processing
                if isinstance(potential_entity_data, Exception):
                    logger.error(f"Step 4a (Entity Types) failed with exception: {potential_entity_data}", exc_info=potential_entity_data)
                    print(f"Error in Step 4a (Entity Types): {type(potential_entity_data).__name__}: {potential_entity_data}")
                    entity_data = None
                elif isinstance(potential_entity_data, EntityTypeSchema) or potential_entity_data is None:
                    entity_data = potential_entity_data
                else:
                    logger.error(f"Step 4a (Entity Types) returned unexpected type: {type(potential_entity_data)}")
                    entity_data = None

                # Ontology Data Processing
                if isinstance(potential_ontology_data, Exception):
                    logger.error(f"Step 4b (Ontology Types) failed with exception: {potential_ontology_data}", exc_info=potential_ontology_data)
                    print(f"Error in Step 4b (Ontology Types): {type(potential_ontology_data).__name__}: {potential_ontology_data}")
                    ontology_data = None
                elif isinstance(potential_ontology_data, OntologyTypeSchema) or potential_ontology_data is None:
                    ontology_data = potential_ontology_data
                else:
                    logger.error(f"Step 4b (Ontology Types) returned unexpected type: {type(potential_ontology_data)}")
                    ontology_data = None

                # Event Data Processing
                if isinstance(potential_event_data, Exception):
                    logger.error(f"Step 4c (Event Types) failed with exception: {potential_event_data}", exc_info=potential_event_data)
                    print(f"Error in Step 4c (Event Types): {type(potential_event_data).__name__}: {potential_event_data}")
                    event_data = None
                elif isinstance(potential_event_data, EventSchema) or potential_event_data is None:
                    event_data = potential_event_data
                else:
                    logger.error(f"Step 4c (Event Types) returned unexpected type: {type(potential_event_data)}")
                    event_data = None

                # Statement Data Processing
                if isinstance(potential_statement_data, Exception):
                    logger.error(f"Step 4d (Statement Types) failed with exception: {potential_statement_data}", exc_info=potential_statement_data)
                    print(f"Error in Step 4d (Statement Types): {type(potential_statement_data).__name__}: {potential_statement_data}")
                    statement_data = None
                elif isinstance(potential_statement_data, StatementTypeSchema) or potential_statement_data is None:
                    statement_data = potential_statement_data
                else:
                    logger.error(f"Step 4d (Statement Types) returned unexpected type: {type(potential_statement_data)}")
                    statement_data = None

                # Evidence Data Processing
                if isinstance(potential_evidence_data, Exception):
                    logger.error(f"Step 4e (Evidence Types) failed with exception: {potential_evidence_data}", exc_info=potential_evidence_data)
                    print(f"Error in Step 4e (Evidence Types): {type(potential_evidence_data).__name__}: {potential_evidence_data}")
                    evidence_data = None
                elif isinstance(potential_evidence_data, EvidenceTypeSchema) or potential_evidence_data is None:
                    evidence_data = potential_evidence_data
                else:
                    logger.error(f"Step 4e (Evidence Types) returned unexpected type: {type(potential_evidence_data)}")
                    evidence_data = None

                # Measurement Data Processing
                if isinstance(potential_measurement_data, Exception):
                    logger.error(f"Step 4f (Measurement Types) failed with exception: {potential_measurement_data}", exc_info=potential_measurement_data)
                    print(f"Error in Step 4f (Measurement Types): {type(potential_measurement_data).__name__}: {potential_measurement_data}")
                    measurement_data = None
                elif isinstance(potential_measurement_data, MeasurementTypeSchema) or potential_measurement_data is None:
                    measurement_data = potential_measurement_data
                else:
                    logger.error(f"Step 4f (Measurement Types) returned unexpected type: {type(potential_measurement_data)}")
                    measurement_data = None

                # Modality Data Processing (New Step 4g)
                if isinstance(potential_modality_data, Exception):
                    logger.error(f"Step 4g (Modality Types) failed with exception: {potential_modality_data}", exc_info=potential_modality_data)
                    print(f"Error in Step 4g (Modality Types): {type(potential_modality_data).__name__}: {potential_modality_data}")
                    modality_data = None
                elif isinstance(potential_modality_data, ModalityTypeSchema) or potential_modality_data is None:
                    modality_data = potential_modality_data
                else:
                    logger.error(f"Step 4g (Modality Types) returned unexpected type: {type(potential_modality_data)}")
                    modality_data = None


                logger.info("--- Finished Step 4: Parallel Identification ---")
                print("--- Finished Step 4: Parallel Identification ---")
            else:
                logger.warning("Skipping Step 4 (Parallel ID) because prerequisites were not met.")
                print("Skipping Step 4 (Parallel ID) due to missing prior results.")


            # === Step 5: Extract Specific Entity Instances ===
            instance_data = await identify_entity_instances(
                content, primary_domain, sub_domain_data, topic_data, entity_data, overall_trace_id
            ) if primary_domain and sub_domain_data and topic_data and entity_data else None

            # === Step 5b: Extract Ontology Concept Instances ===
            ontology_instance_data = await identify_ontology_instances(
                content, primary_domain, sub_domain_data, topic_data, ontology_data, overall_trace_id
            ) if primary_domain and sub_domain_data and topic_data and ontology_data else None

            # === Step 5c: Extract Event Instances ===
            event_instance_data = await identify_event_instances(
                content, primary_domain, sub_domain_data, topic_data, event_data, overall_trace_id
            ) if primary_domain and sub_domain_data and topic_data and event_data else None

            # === Step 5d: Extract Statement Instances ===
            statement_instance_data = await identify_statement_instances(
                content, primary_domain, sub_domain_data, topic_data, statement_data, overall_trace_id
            ) if primary_domain and sub_domain_data and topic_data and statement_data else None

            # === Step 5e: Extract Evidence Instances ===
            evidence_instance_data = await identify_evidence_instances(
                content, primary_domain, sub_domain_data, topic_data, evidence_data, overall_trace_id
            ) if primary_domain and sub_domain_data and topic_data and evidence_data else None

            # === Step 5f: Extract Measurement Instances ===
            measurement_instance_data = await identify_measurement_instances(
                content, primary_domain, sub_domain_data, topic_data, measurement_data, overall_trace_id
            ) if primary_domain and sub_domain_data and topic_data and measurement_data else None

            # === Step 5g: Extract Modality Instances ===
            modality_instance_data = await identify_modality_instances(
                content, primary_domain, sub_domain_data, topic_data, modality_data, overall_trace_id
            ) if primary_domain and sub_domain_data and topic_data and modality_data else None

            # === Step 6: Identify Relationships in PARALLEL for each Entity Type (Based on Context) ===
            # Note: This step currently only uses entity_data. If relationships involving other types were needed,
            # the step would require modification to accept and use that data.
            relationship_data = await identify_relationship_types(
                content, primary_domain, sub_domain_data, topic_data, entity_data, overall_trace_id
            ) if primary_domain and sub_domain_data and topic_data and entity_data else None


            # Log completion status of individual steps (optional)
            logger.info(f"Step 1 (Domain) Result: {'Success' if domain_data else 'Failed/Skipped'}")
            logger.info(f"Step 2 (SubDomain) Result: {'Success' if sub_domain_data else 'Failed/Skipped'}")
            logger.info(f"Step 3 (Topics) Result: {'Success' if topic_data else 'Failed/Skipped'}")
            logger.info(f"Step 4a (Entity Types) Result: {'Success' if entity_data else 'Failed/Skipped/Error'}")
            logger.info(f"Step 4b (Ontology Types) Result: {'Success' if ontology_data else 'Failed/Skipped/Error'}")
            logger.info(f"Step 4c (Event Types) Result: {'Success' if event_data else 'Failed/Skipped/Error'}")
            logger.info(f"Step 4d (Statement Types) Result: {'Success' if statement_data else 'Failed/Skipped/Error'}")
            logger.info(f"Step 4e (Evidence Types) Result: {'Success' if evidence_data else 'Failed/Skipped/Error'}")
            logger.info(f"Step 4f (Measurement Types) Result: {'Success' if measurement_data else 'Failed/Skipped/Error'}")
            logger.info(f"Step 4g (Modality Types) Result: {'Success' if modality_data else 'Failed/Skipped/Error'}") # Added log for new step (4g)
            logger.info(f"Step 5 (Entity Instances) Result: {'Success' if instance_data else 'Failed/Skipped'}")
            logger.info(f"Step 5b (Ontology Instances) Result: {'Success' if ontology_instance_data else 'Failed/Skipped'}")
            logger.info(f"Step 5c (Event Instances) Result: {'Success' if event_instance_data else 'Failed/Skipped'}")
            logger.info(f"Step 5d (Statement Instances) Result: {'Success' if statement_instance_data else 'Failed/Skipped'}")
            logger.info(f"Step 5e (Evidence Instances) Result: {'Success' if evidence_instance_data else 'Failed/Skipped'}")
            logger.info(f"Step 5f (Measurement Instances) Result: {'Success' if measurement_instance_data else 'Failed/Skipped'}")
            logger.info(f"Step 5g (Modality Instances) Result: {'Success' if modality_instance_data else 'Failed/Skipped'}")
            logger.info(f"Step 6 (Relationships) Result: {'Success' if relationship_data else 'Failed/Skipped'}")


    except Exception as e:
        # Catch errors occurring outside the specific steps but within the trace
        logger.exception("An unexpected error occurred within the main workflow trace.", extra={"trace_id": overall_trace_id or 'N/A'})
        print(f"\nAn unexpected error occurred during the main workflow: {type(e).__name__}: {e}")

    # This message prints regardless of success/failure within the trace
    print(f"\nFull Workflow ({overall_trace_metadata['workflow_name']}) finished.")
    logger.info(f"--- Finished Analysis Workflow Trace ({overall_trace_metadata['workflow_name']}) (ID: {overall_trace_id or 'N/A'}) ---")