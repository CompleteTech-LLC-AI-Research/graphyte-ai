"""Step 5a: Entity instance extraction functionality."""

import logging
from datetime import datetime, timezone
from typing import List, Optional, cast

from pydantic import ValidationError

from agents import RunConfig, RunResult, TResponseInputItem  # type: ignore[attr-defined]

from ..workflow_agents import entity_instance_extractor_agent
from ..config import (
    ENTITY_INSTANCE_MODEL,
    ENTITY_INSTANCE_OUTPUT_DIR,
    ENTITY_INSTANCE_OUTPUT_FILENAME,
)
from ..schemas import (
    EntityInstanceSchema,
    EntityInstanceBaseSchema,
    SubDomainSchema,
    TopicSchema,
    EntityTypeSchema,
)
from ..utils import (
    direct_save_json_output,
    run_agent_with_retry,
    score_entity_instances,
)

logger = logging.getLogger(__name__)


async def identify_entity_instances(
    content: str,
    primary_domain: str,
    sub_domain_data: SubDomainSchema,
    topic_data: TopicSchema,
    entity_data: EntityTypeSchema,
    trace_id: Optional[str] = None,
    group_id: Optional[str] = None,
) -> Optional[EntityInstanceSchema]:
    """Extract specific entity mentions from the text based on context."""
    if not primary_domain or not sub_domain_data or not topic_data or not entity_data:
        logger.info("Skipping Step 5a because prerequisites were not identified.")
        if not primary_domain:
            print("Skipping Step 5a as primary domain was not identified.")
        elif not sub_domain_data:
            print("Skipping Step 5a as sub-domain identification failed.")
        elif not topic_data:
            print("Skipping Step 5a as topic identification failed.")
        elif not entity_data:
            print("Skipping Step 5a as entity type identification failed.")
        return None

    logger.info(
        f"--- Running Step 5a: Entity Instance Extraction (Agent: {entity_instance_extractor_agent.name}) ---"
    )
    print(
        f"\n--- Running Step 5a: Entity Instance Extraction using model: {ENTITY_INSTANCE_MODEL} ---"
    )

    step5a_metadata_for_trace = {
        "workflow_step": "5a_entity_instance_extraction",
        "agent_name": "Entity Instance Extractor",
        "actual_agent": str(entity_instance_extractor_agent.name),
        "primary_domain_input": primary_domain,
        "sub_domains_analyzed_count": str(len(sub_domain_data.identified_sub_domains)),
        "topic_context_count": str(
            sum(len(t.identified_topics) for t in topic_data.sub_domain_topic_map)
        ),
        "entity_type_count": str(len(entity_data.identified_entities)),
    }
    step5a_run_config = RunConfig(
        workflow_name="step5a_entity_instances",
        trace_id=trace_id,
        group_id=group_id,
        trace_metadata={k: str(v) for k, v in step5a_metadata_for_trace.items()},
    )
    step5a_result: Optional[RunResult] = None
    instance_data: Optional[EntityInstanceBaseSchema | EntityInstanceSchema] = None

    context_summary_for_prompt = (
        f"Primary Domain: {primary_domain}\n"
        f"Identified Sub-Domains: {', '.join(sd.sub_domain for sd in sub_domain_data.identified_sub_domains)}\n"
        f"Entity Types Considered: {', '.join(et.entity_type for et in entity_data.identified_entities)}"
    )

    step5a_input_list: List[TResponseInputItem] = [
        {
            "role": "user",
            "content": (
                f"Extract specific entity mentions from the text. Use the provided context:\n{context_summary_for_prompt}\n\n"
                f"Provide the entity type, exact text span and character offsets. Output ONLY using the required EntityInstanceSchema."
            ),
        },
        {
            "role": "user",
            "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---",
        },
    ]

    try:
        step5a_result = await run_agent_with_retry(
            agent=entity_instance_extractor_agent,
            input_data=step5a_input_list,
            config=step5a_run_config,
        )

        if step5a_result:
            potential_output = getattr(step5a_result, "final_output", None)
            if isinstance(potential_output, EntityInstanceBaseSchema):
                instance_data = potential_output
            elif isinstance(potential_output, dict):
                try:
                    instance_data = EntityInstanceBaseSchema.model_validate(
                        potential_output
                    )
                except ValidationError as e:
                    logger.warning(
                        f"Step 5a dict output failed EntityInstanceBaseSchema validation: {e}"
                    )
            else:
                logger.warning(
                    f"Step 5a final_output was not EntityInstanceBaseSchema or dict (type: {type(potential_output)})."
                )

            if instance_data and instance_data.identified_instances:
                if instance_data.primary_domain != primary_domain:
                    instance_data.primary_domain = primary_domain
                if not set(instance_data.analyzed_sub_domains):
                    instance_data.analyzed_sub_domains = [
                        sd.sub_domain for sd in sub_domain_data.identified_sub_domains
                    ]
                instance_data = await score_entity_instances(instance_data, content)
                logger.info(
                    f"Step 5a Result (Structured Instances):\n{instance_data.model_dump_json(indent=2)}"
                )
                print("\n--- Entity Instances Extracted (Structured Output) ---")
                print(instance_data.model_dump_json(indent=2))

                output_content = {
                    "primary_domain": instance_data.primary_domain,
                    "analyzed_sub_domains": instance_data.analyzed_sub_domains,
                    "analyzed_entity_types": instance_data.analyzed_entity_types,
                    "identified_instances": [
                        item.model_dump() for item in instance_data.identified_instances
                    ],
                    "analysis_summary": instance_data.analysis_summary,
                    "analysis_details": {
                        "source_text_length": len(content),
                        "model_used": ENTITY_INSTANCE_MODEL,
                        "agent_name": entity_instance_extractor_agent.name,
                        "output_schema": EntityInstanceSchema.__name__,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    "trace_information": {
                        "trace_id": trace_id or "N/A",
                        "notes": f"Generated by {entity_instance_extractor_agent.name} in Step 5a of workflow.",
                    },
                }
                save_result = direct_save_json_output(
                    ENTITY_INSTANCE_OUTPUT_DIR,
                    ENTITY_INSTANCE_OUTPUT_FILENAME,
                    output_content,
                    trace_id,
                )
                print(f"  - {save_result}")
                logger.info(f"Result of saving entity instance output: {save_result}")
            elif instance_data and not instance_data.identified_instances:
                logger.warning(
                    "Step 5a completed but identified_instances list is empty."
                )
                print("\nStep 5a completed, but no entity instances were identified.")
            else:
                logger.error(
                    "Step 5a FAILED: Could not extract valid EntityInstanceSchema output."
                )
                print("\nError: Failed to extract entity instances in Step 5a.")
                instance_data = None
        else:
            logger.error("Step 5a FAILED: Runner.run did not return a result.")
            print(
                "\nError: Failed to get a result from the entity instance extraction step."
            )
            instance_data = None

    except (ValidationError, TypeError) as e:
        logger.exception(
            f"Validation or Type error during Step 5a agent run. Error: {e}",
            extra={"trace_id": trace_id or "N/A"},
        )
        print("\nError: A data validation or type issue occurred during Step 5a.")
        print(f"Error details: {e}")
        instance_data = None
    except Exception as e:
        logger.exception(
            "An unexpected error occurred during Step 5a.",
            extra={"trace_id": trace_id or "N/A"},
        )
        print(f"\nAn unexpected error occurred during Step 5a: {type(e).__name__}: {e}")
        instance_data = None

    return cast(Optional[EntityInstanceSchema], instance_data)
