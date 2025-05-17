"""Step 5: Relationship type identification functionality."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import ValidationError

from agentic_team import RunConfig, RunResult, TResponseInputItem

from ..agents import relationship_type_identifier_agent
from ..config import (
    RELATIONSHIP_MODEL,
    RELATIONSHIP_OUTPUT_DIR,
    RELATIONSHIP_OUTPUT_FILENAME,
)
from ..schemas import (
    RelationshipSchema,
    SingleEntityTypeRelationshipSchema,
    SubDomainSchema,
    TopicSchema,
    EntityTypeSchema,
)
from ..utils import direct_save_json_output, run_agent_with_retry

logger = logging.getLogger(__name__)


async def identify_relationship_types(
    content: str,
    primary_domain: str,
    sub_domain_data: SubDomainSchema,
    topic_data: TopicSchema,
    entity_data: EntityTypeSchema,
    overall_trace_id: Optional[str] = None,
) -> Optional[RelationshipSchema]:
    """Identify relationship types for each entity type focus.

    Args:
        content: The text content to analyze
        primary_domain: The primary domain identified in step 1
        sub_domain_data: The SubDomainSchema from step 2
        topic_data: The TopicSchema from step 3
        entity_data: The EntityTypeSchema from step 4a
        overall_trace_id: The trace ID for logging purposes

    Returns:
        A RelationshipSchema object if successful, None otherwise
    """
    if not primary_domain or not sub_domain_data or not topic_data or not entity_data:
        logger.info("Skipping Step 5 because prerequisites were not identified.")
        if not primary_domain:
            print("Skipping Step 5 as primary domain was not identified.")
        elif not sub_domain_data:
            print("Skipping Step 5 as sub-domain identification failed.")
        elif not topic_data:
            print("Skipping Step 5 as topic identification failed.")
        elif not entity_data:
            print("Skipping Step 5 as entity type identification failed.")
        return None

    # Extract unique entity types identified in Step 4a
    entity_types_list_for_step5 = sorted(
        list(
            set(
                item.entity_type.strip()
                for item in entity_data.identified_entities
                if item.entity_type and item.entity_type.strip()
            )
        )
    )

    if not entity_types_list_for_step5:
        logger.warning(
            "Step 4a identified an entity list, but it's empty after filtering/stripping. Skipping Step 5."
        )
        print(
            "\nStep 4a completed, but no specific non-empty entity types were identified. Cannot proceed to relationship analysis."
        )
        return None

    logger.info(
        f"--- Starting Step 5: PARALLEL Relationship ID (Agent: {relationship_type_identifier_agent.name}) for {len(entity_types_list_for_step5)} Entity Type(s) ---"
    )
    print(
        f"\n--- Running Step 5: PARALLEL Relationship ID using model: {RELATIONSHIP_MODEL} ---"
    )

    relationship_tasks = []
    entity_types_being_processed = (
        []
    )  # Track which entity type corresponds to which task/result
    aggregated_relationship_results: List[SingleEntityTypeRelationshipSchema] = []
    relationship_data: Optional[RelationshipSchema] = None

    # --- Prepare context summary for relationship agent prompt ---
    # More detailed than step 4, potentially summarizing topics per sub-domain briefly
    context_summary_for_relation_prompt = (
        f"Overall Context:\n"
        f"- Primary Domain: {primary_domain}\n"
        f"- Identified Sub-Domains: {', '.join(sd.sub_domain for sd in sub_domain_data.identified_sub_domains)}\n"
        f"- Identified Entity Types: {', '.join(et.entity_type for et in entity_data.identified_entities)}\n"
        f"- Relevant Topics Found:\n"
    )
    topic_summary_lines = []
    for topic_map in topic_data.sub_domain_topic_map:
        top_topics = [
            t.topic for t in topic_map.identified_topics[:3]
        ]  # Top 3 topics per sub-domain
        if top_topics:
            topic_summary_lines.append(
                f"  - For '{topic_map.sub_domain}': {', '.join(top_topics)}{'...' if len(topic_map.identified_topics) > 3 else ''}"
            )
    if not topic_summary_lines:
        topic_summary_lines.append("  - (No specific topics successfully extracted)")

    context_summary_for_relation_prompt += "\n".join(topic_summary_lines)

    # --- Prepare tasks for parallel execution ---
    for index, current_entity_type in enumerate(entity_types_list_for_step5):
        logger.debug(
            f"Preparing task for Step 5 ({index+1}/{len(entity_types_list_for_step5)}): Entity Type Focus '{current_entity_type}'"
        )

        display_entity_type = (
            (current_entity_type[:25] + "...")
            if len(current_entity_type) > 28
            else current_entity_type
        )
        step5_iter_metadata_for_trace = {
            "workflow_step": f"5_relationship_id_batch_{index+1}",
            "agent_name": f"Relationship ID ({display_entity_type})",
            "actual_agent": str(relationship_type_identifier_agent.name),
            "primary_domain_input": primary_domain,
            "entity_type_focus": current_entity_type,
            "batch_index": str(index + 1),
            "batch_size": str(len(entity_types_list_for_step5)),
            "context_subdomain_count": str(len(sub_domain_data.identified_sub_domains)),
            "context_topic_count": str(
                sum(len(t.identified_topics) for t in topic_data.sub_domain_topic_map)
            ),
            "context_entity_type_count": str(len(entity_data.identified_entities)),
        }
        step5_iter_run_config = RunConfig(
            trace_metadata={k: str(v) for k, v in step5_iter_metadata_for_trace.items()}
        )

        step5_iter_input_list: List[TResponseInputItem] = [
            {
                "role": "user",
                "content": (
                    f"Analyze the relationships in the following text. Focus specifically on relationships involving the entity type: **'{current_entity_type}'**. "
                    f"Use the provided context:\n{context_summary_for_relation_prompt}\n\n"
                    f"Identify relationships where '{current_entity_type}' is one of the participants, providing details (entity types, names if possible, relationship type, score). "
                    f"Output ONLY using the required SingleEntityTypeRelationshipSchema, ensuring 'entity_type_focus' is '{current_entity_type}'."
                ),
            },
            {
                "role": "user",
                "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---",
            },
        ]

        # Create the async task
        task = asyncio.create_task(
            run_agent_with_retry(
                agent=relationship_type_identifier_agent,
                input_data=step5_iter_input_list,
                config=step5_iter_run_config,
            ),
            name=f"RelTask_{current_entity_type[:20]}",
        )
        relationship_tasks.append(task)
        entity_types_being_processed.append(current_entity_type)

    # --- Execute tasks in parallel ---
    if (
        not relationship_tasks
    ):  # Should not happen if entity_types_list_for_step5 was populated, but safeguard
        logger.warning(
            "No relationship identification tasks were prepared in Step 5. Skipping."
        )
        print(
            "No relationship identification tasks prepared. Skipping Step 5 execution."
        )
        return None

    logger.info(
        f"Launching {len(relationship_tasks)} relationship identification tasks in parallel..."
    )
    print(
        f"Running relationship identification for {len(relationship_tasks)} entity types concurrently..."
    )
    step5_results_list = await asyncio.gather(
        *relationship_tasks, return_exceptions=True
    )
    logger.info("Parallel relationship identification tasks completed.")
    print("Parallel relationship identification runs finished. Processing results...")

    # --- Process results from parallel execution ---
    for index, step5_iter_result_or_exc in enumerate(step5_results_list):
        current_entity_type = entity_types_being_processed[index]

        try:
            # Handle exceptions from gather
            if isinstance(step5_iter_result_or_exc, Exception):
                logger.error(
                    f"Step 5 task for focus '{current_entity_type}' failed with exception: {step5_iter_result_or_exc}",
                    exc_info=step5_iter_result_or_exc,
                )
                print(
                    f"  - Error processing relationships for focus '{current_entity_type}': {type(step5_iter_result_or_exc).__name__}: {step5_iter_result_or_exc}"
                )
                continue

            # Process successful result
            step5_iter_result: Optional[RunResult] = step5_iter_result_or_exc
            if step5_iter_result:
                potential_output_iter = getattr(step5_iter_result, "final_output", None)
                single_relation_data: Optional[SingleEntityTypeRelationshipSchema] = (
                    None
                )

                if isinstance(
                    potential_output_iter, SingleEntityTypeRelationshipSchema
                ):
                    single_relation_data = potential_output_iter
                    logger.info(
                        f"Successfully extracted SingleEntityTypeRelationshipSchema for focus '{current_entity_type}'."
                    )
                elif isinstance(potential_output_iter, dict):
                    try:
                        single_relation_data = (
                            SingleEntityTypeRelationshipSchema.model_validate(
                                potential_output_iter
                            )
                        )
                        logger.info(
                            f"Successfully validated SingleEntityTypeRelationshipSchema from dict for focus '{current_entity_type}'."
                        )
                    except ValidationError as e:
                        logger.warning(
                            f"Dict output for focus '{current_entity_type}' failed SingleEntityTypeRelationshipSchema validation: {e}"
                        )
                else:
                    logger.warning(
                        f"Output for focus '{current_entity_type}' was not SingleEntityTypeRelationshipSchema or dict (type: {type(potential_output_iter)}). Raw: {potential_output_iter}"
                    )

                if single_relation_data:
                    # Ensure focus type matches
                    if (
                        single_relation_data.entity_type_focus.strip().upper()
                        != current_entity_type.strip().upper()
                    ):
                        logger.warning(
                            f"Entity type focus mismatch in output for '{current_entity_type}'. Output had '{single_relation_data.entity_type_focus}'. Correcting."
                        )
                        single_relation_data.entity_type_focus = current_entity_type

                    # Log and print details for this focus type
                    relation_details = single_relation_data.identified_relationships
                    logger.info(
                        f"Step 5 Result for focus '{current_entity_type}': Found {len(relation_details)} relationships."
                    )
                    print(
                        f"\n  --- Relationships involving Focus Type: '{current_entity_type}' ---"
                    )
                    if relation_details:
                        for rel in relation_details:
                            print(f"     - {rel.relationship_type}")
                    else:
                        print(
                            "     - (No specific relationships identified for this focus type)"
                        )

                    # Add successfully processed result
                    aggregated_relationship_results.append(single_relation_data)
                else:
                    logger.warning(
                        f"Could not extract valid relationship data for focus '{current_entity_type}'. Raw output: {potential_output_iter}"
                    )
                    print(
                        f"  - Warning: Failed to get structured relationships for focus '{current_entity_type}'."
                    )
            else:
                logger.error(
                    f"Step 5 task for focus '{current_entity_type}' returned no result object."
                )
                print(
                    f"  - Error: Failed to get result object for relationship focus '{current_entity_type}'."
                )

        except (ValidationError, TypeError) as e:
            logger.exception(
                f"Validation or Type error processing relationship result for focus '{current_entity_type}'. Error: {e}",
                extra={"trace_id": overall_trace_id or "N/A"},
            )
            print(
                f"\nError: A data validation or type issue occurred processing relationship result for focus '{current_entity_type}'."
            )
            print(f"Error details: {e}")
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred processing relationship result for focus '{current_entity_type}'.",
                extra={"trace_id": overall_trace_id or "N/A"},
            )
            print(
                f"\nAn unexpected error occurred processing relationship result for focus '{current_entity_type}': {type(e).__name__}: {e}"
            )
    # --- End of processing loop for parallel relationship results ---

    # === After Parallel Runs: Aggregate and Save Final Relationship Output ===
    if not aggregated_relationship_results:
        logger.warning(
            "Step 5 (Parallel Relationship ID) completed, but no relationship results were successfully aggregated. Final relationship file not saved."
        )
        print(
            "\nStep 5 (Parallel) completed, but no relationship results were successfully aggregated. Final relationship file not saved."
        )
        return None

    logger.info(
        "Aggregating relationship results from parallel runs and saving final output."
    )
    print(
        "\n--- Aggregating and Saving Final Relationship Analysis (from parallel runs) ---"
    )

    # Store final aggregated data
    relationship_data = RelationshipSchema(
        primary_domain=primary_domain,
        analyzed_sub_domains=[
            sd.sub_domain for sd in sub_domain_data.identified_sub_domains
        ],  # Use list from step 2
        analyzed_entity_types=entity_types_being_processed,  # List of types attempted
        entity_relationships_map=aggregated_relationship_results,  # List of successful results
        analysis_summary=f"Generated relationships in parallel focusing on {len(aggregated_relationship_results)} entity types (out of {len(entity_types_being_processed)} attempted).",
    )

    logger.info(
        f"Final Aggregated Relationships (Structured):\n{relationship_data.model_dump_json(indent=2)}"
    )
    print(
        "\n--- Final Aggregated Relationships (Structured Output from Step 5 Parallel Runs) ---"
    )
    print(relationship_data.model_dump_json(indent=2))

    relationship_output_content = {
        "primary_domain": relationship_data.primary_domain,
        "analyzed_sub_domains": relationship_data.analyzed_sub_domains,
        "analyzed_entity_types": relationship_data.analyzed_entity_types,
        "entity_relationships_map": [
            item.model_dump() for item in relationship_data.entity_relationships_map
        ],
        "analysis_summary": relationship_data.analysis_summary,
        "analysis_details": {
            "source_text_length": len(content),
            "primary_domain_context": primary_domain,
            "sub_domain_context_count": len(sub_domain_data.identified_sub_domains),
            "topic_context_count": sum(
                len(t.identified_topics) for t in topic_data.sub_domain_topic_map
            ),
            "entity_type_context_count": len(entity_data.identified_entities),
            "entity_types_focused_on": entity_types_being_processed,
            "entity_types_successfully_processed_count": len(
                aggregated_relationship_results
            ),
            "execution_mode": "Parallel (asyncio.gather)",
            "model_used_per_relationship_call": RELATIONSHIP_MODEL,
            "agent_name_per_relationship_call": relationship_type_identifier_agent.name,
            "output_schema_final": RelationshipSchema.__name__,
            "output_schema_per_call": SingleEntityTypeRelationshipSchema.__name__,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "trace_information": {
            "trace_id": overall_trace_id or "N/A",
            "notes": f"Aggregated from PARALLEL calls to {relationship_type_identifier_agent.name} in Step 5 of workflow.",
        },
    }
    save_result_step5_final = direct_save_json_output(
        RELATIONSHIP_OUTPUT_DIR,
        RELATIONSHIP_OUTPUT_FILENAME,
        relationship_output_content,
        overall_trace_id,
    )
    print("\nSaving final aggregated relationship output file...")
    print(f"  - {save_result_step5_final}")
    logger.info(
        f"Result of saving final aggregated relationship output: {save_result_step5_final}"
    )

    return relationship_data
