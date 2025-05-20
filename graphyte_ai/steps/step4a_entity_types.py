"""Step 4a: Entity type identification functionality."""

import logging
from datetime import datetime, timezone
from typing import List, Optional, cast

from pydantic import ValidationError

from agents import RunConfig, RunResult, TResponseInputItem  # type: ignore[attr-defined]

from ..workflow_agents import entity_type_identifier_agent
from ..config import (
    ENTITY_TYPE_MODEL,
    ENTITY_TYPE_OUTPUT_DIR,
    ENTITY_TYPE_OUTPUT_FILENAME,
)
from ..schemas import (
    SubDomainSchema,
    TopicSchema,
    EntityTypeSchema,
    EntityTypeBaseSchema,
)
from ..utils import (
    direct_save_json_output,
    run_agent_with_retry,
    score_entity_types,
)

logger = logging.getLogger(__name__)


async def identify_entity_types(
    content: str,
    primary_domain: str,
    sub_domain_data: SubDomainSchema,
    topic_data: TopicSchema,
    trace_id: Optional[str] = None,
    group_id: Optional[str] = None,
) -> Optional[EntityTypeSchema]:
    """Identify entity types based on domain, sub-domains, and topics.

    Args:
        content: The text content to analyze
        primary_domain: The primary domain identified in step 1
        sub_domain_data: The SubDomainSchema from step 2
        topic_data: The TopicSchema from step 3
        trace_id: The trace ID for logging purposes
        group_id: The trace group ID for logging purposes

    Returns:
        An EntityTypeSchema object if successful, None otherwise
    """
    if not primary_domain or not sub_domain_data or not topic_data:
        logger.info("Skipping Step 4a because prerequisites were not identified.")
        if not primary_domain:
            print("Skipping Step 4a as primary domain was not identified.")
        elif not sub_domain_data:
            print("Skipping Step 4a as sub-domain identification failed.")
        elif not topic_data:
            print("Skipping Step 4a as topic identification failed.")
        return None

    logger.info(
        f"--- Running Step 4a: Entity Type ID (Agent: {entity_type_identifier_agent.name}) ---"
    )
    print(f"\n--- Running Step 4a: Entity Type ID using model: {ENTITY_TYPE_MODEL} ---")

    step4_metadata_for_trace = {
        "workflow_step": "4a_entity_type_id",
        "agent_name": "Entity Type ID",
        "actual_agent": str(entity_type_identifier_agent.name),
        "primary_domain_input": primary_domain,
        "sub_domains_analyzed_count": str(len(sub_domain_data.identified_sub_domains)),
        "topics_aggregated_count": str(
            sum(len(t.identified_topics) for t in topic_data.sub_domain_topic_map)
        ),
    }
    step4_run_config = RunConfig(
        workflow_name="step4a_entity_types",
        trace_id=trace_id,
        group_id=group_id,
        trace_metadata={k: str(v) for k, v in step4_metadata_for_trace.items()},
    )
    step4_result: Optional[RunResult] = None
    entity_data: Optional[EntityTypeBaseSchema | EntityTypeSchema] = None

    # Prepare context summary for the prompt
    context_summary_for_prompt = (
        f"Primary Domain: {primary_domain}\n"
        f"Identified Sub-Domains: {', '.join(sd.sub_domain for sd in sub_domain_data.identified_sub_domains)}\n"
        f"Previously identified topics (aggregated): {len(topic_data.sub_domain_topic_map)} sub-domains covered with topics."
    )

    step4_input_list: List[TResponseInputItem] = [
        {
            "role": "user",
            "content": (
                f"Analyze the following text to identify key entity types (e.g., PERSON, ORGANIZATION, LOCATION, DATE, "
                f"MONEY, PRODUCT, EVENT, TECHNOLOGY, SCIENTIFIC_CONCEPT, ECONOMIC_INDICATOR). "
                f"Use the provided context:\n{context_summary_for_prompt}\n\n"
                f"Identify entity types relevant to this overall context. "
                f"Output ONLY using the required EntityTypeSchema, including the primary_domain and analyzed_sub_domains list in the output."
            ),
        },
        {
            "role": "user",
            "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---",
        },
    ]

    try:
        step4_result = await run_agent_with_retry(
            agent=entity_type_identifier_agent,
            input_data=step4_input_list,
            config=step4_run_config,
        )

        if step4_result:
            potential_output_step4 = getattr(step4_result, "final_output", None)
            if isinstance(potential_output_step4, EntityTypeBaseSchema):
                entity_data = potential_output_step4
                logger.info(
                    "Successfully extracted EntityTypeBaseSchema from step4_result.final_output."
                )
            elif isinstance(potential_output_step4, dict):
                try:
                    entity_data = EntityTypeBaseSchema.model_validate(
                        potential_output_step4
                    )
                    logger.info(
                        "Successfully validated EntityTypeBaseSchema from step4_result.final_output dict."
                    )
                except ValidationError as e:
                    logger.warning(
                        f"Step 4a dict output failed EntityTypeBaseSchema validation: {e}"
                    )
            else:
                logger.warning(
                    f"Step 4a final_output was not EntityTypeBaseSchema or dict (type: {type(potential_output_step4)})."
                )

            if entity_data and entity_data.identified_entities:
                # Ensure context fields in output schema match expectations
                if entity_data.primary_domain != primary_domain:
                    logger.warning(
                        f"Primary domain mismatch in Step 4a output ('{entity_data.primary_domain}'). Overwriting with Step 1's ('{primary_domain}')."
                    )
                    entity_data.primary_domain = primary_domain

                if set(entity_data.analyzed_sub_domains) != set(
                    sd.sub_domain for sd in sub_domain_data.identified_sub_domains
                ):
                    logger.warning(
                        f"Analyzed sub-domains in Step 4a output {entity_data.analyzed_sub_domains} differs from Step 2 input { [sd.sub_domain for sd in sub_domain_data.identified_sub_domains]}. Using Step 4a's list."
                    )

                scored_entity_data = await score_entity_types(entity_data, content)
                entity_data = scored_entity_data

                # Log and print results
                entity_log_items = [
                    item.entity_type for item in entity_data.identified_entities
                ]
                logger.info(
                    f"Step 4a Result: Identified Entity Types = [{', '.join(entity_log_items)}]"
                )
                logger.info(
                    f"Step 4a Result (Structured Entities):\n{entity_data.model_dump_json(indent=2)}"
                )
                print(
                    "\n--- Entity Types Identified (Structured Output from Step 4a) ---"
                )
                print(entity_data.model_dump_json(indent=2))

                # Save results
                logger.info("Saving entity type identifier output to file...")
                print("\nSaving entity type output file...")
                entity_type_output_content = {
                    "primary_domain": entity_data.primary_domain,
                    "analyzed_sub_domains": entity_data.analyzed_sub_domains,
                    "identified_entities": [
                        item.model_dump() for item in entity_data.identified_entities
                    ],
                    "sub_domain_entity_map": [
                        item.model_dump() for item in entity_data.sub_domain_entity_map
                    ],
                    "analysis_summary": entity_data.analysis_summary,
                    "analysis_details": {
                        "source_text_length": len(content),
                        "primary_domain_context": primary_domain,
                        "sub_domain_context_count": len(
                            sub_domain_data.identified_sub_domains
                        ),
                        "topic_context_count": sum(
                            len(t.identified_topics)
                            for t in topic_data.sub_domain_topic_map
                        ),
                        "model_used": ENTITY_TYPE_MODEL,
                        "agent_name": entity_type_identifier_agent.name,
                        "output_schema": EntityTypeSchema.__name__,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    "trace_information": {
                        "trace_id": trace_id or "N/A",
                        "notes": f"Generated by {entity_type_identifier_agent.name} in Step 4a of workflow.",
                    },
                }
                save_result_step4 = direct_save_json_output(
                    ENTITY_TYPE_OUTPUT_DIR,
                    ENTITY_TYPE_OUTPUT_FILENAME,
                    entity_type_output_content,
                    trace_id,
                )
                print(f"  - {save_result_step4}")
                logger.info(f"Result of saving entity type output: {save_result_step4}")

            elif entity_data and not entity_data.identified_entities:
                logger.warning(
                    "Step 4a completed but identified_entities list is empty."
                )
                print(
                    "\nStep 4a completed, but no specific entity types were identified."
                )
                entity_data = None  # Signal failure for next step
            else:  # entity_data is None or validation failed
                logger.error(
                    "Step 4a FAILED: Could not extract valid EntityTypeSchema output."
                )
                print("\nError: Failed to identify entity types in Step 4a.")
                entity_data = None  # Signal failure for next step

        else:
            logger.error("Step 4a FAILED: Runner.run did not return a result.")
            print(
                "\nError: Failed to get a result from the entity type identification step."
            )
            entity_data = None

    except (ValidationError, TypeError) as e:
        logger.exception(
            f"Validation or Type error during Step 4a agent run. Error: {e}",
            extra={"trace_id": trace_id or "N/A"},
        )
        print("\nError: A data validation or type issue occurred during Step 4a.")
        print(f"Error details: {e}")
        entity_data = None
    except Exception as e:
        logger.exception(
            "An unexpected error occurred during Step 4a.",
            extra={"trace_id": trace_id or "N/A"},
        )
        print(f"\nAn unexpected error occurred during Step 4a: {type(e).__name__}: {e}")
        entity_data = None

    return cast(Optional[EntityTypeSchema], entity_data)
