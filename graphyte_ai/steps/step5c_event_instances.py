"""Step 5c: Event instance extraction functionality."""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import ValidationError

from agents import RunConfig, RunResult, TResponseInputItem  # type: ignore[attr-defined]

from ..workflow_agents import event_instance_extractor_agent
from ..config import (
    EVENT_INSTANCE_MODEL,
    EVENT_INSTANCE_OUTPUT_DIR,
    EVENT_INSTANCE_OUTPUT_FILENAME,
)
from ..schemas import EventInstanceSchema, SubDomainSchema, TopicSchema, EventTypeSchema
from ..utils import direct_save_json_output, run_agent_with_retry

logger = logging.getLogger(__name__)


async def identify_event_instances(
    content: str,
    primary_domain: str,
    sub_domain_data: SubDomainSchema,
    topic_data: TopicSchema,
    event_data: EventTypeSchema,
    overall_trace_id: Optional[str] = None,
) -> Optional[EventInstanceSchema]:
    """Extract specific event mentions from the text based on context."""
    if not primary_domain or not sub_domain_data or not topic_data or not event_data:
        logger.info("Skipping Step 5c because prerequisites were not identified.")
        if not primary_domain:
            print("Skipping Step 5c as primary domain was not identified.")
        elif not sub_domain_data:
            print("Skipping Step 5c as sub-domain identification failed.")
        elif not topic_data:
            print("Skipping Step 5c as topic identification failed.")
        elif not event_data:
            print("Skipping Step 5c as event type identification failed.")
        return None

    logger.info(
        f"--- Running Step 5c: Event Instance Extraction (Agent: {event_instance_extractor_agent.name}) ---"
    )
    print(
        f"\n--- Running Step 5c: Event Instance Extraction using model: {EVENT_INSTANCE_MODEL} ---"
    )

    step5c_metadata_for_trace = {
        "workflow_step": "5c_event_instance_extraction",
        "agent_name": "Event Instance Extractor",
        "actual_agent": str(event_instance_extractor_agent.name),
        "primary_domain_input": primary_domain,
        "sub_domains_analyzed_count": str(len(sub_domain_data.identified_sub_domains)),
        "topic_context_count": str(
            sum(len(t.identified_topics) for t in topic_data.sub_domain_topic_map)
        ),
        "event_type_count": str(len(event_data.identified_events)),
    }
    step5c_run_config = RunConfig(
        trace_metadata={k: str(v) for k, v in step5c_metadata_for_trace.items()}
    )
    step5c_result: Optional[RunResult] = None
    instance_data: Optional[EventInstanceSchema] = None

    context_summary_for_prompt = (
        f"Primary Domain: {primary_domain}\n"
        f"Identified Sub-Domains: {', '.join(sd.sub_domain for sd in sub_domain_data.identified_sub_domains)}\n"
        f"Event Types Considered: {', '.join(e.event_type for e in event_data.identified_events)}"
    )

    step5c_input_list: List[TResponseInputItem] = [
        {
            "role": "user",
            "content": (
                f"Extract specific mentions of events from the text. Use the provided context:\n{context_summary_for_prompt}\n\n"
                f"Provide the event type, exact text span and character offsets. Output ONLY using the required EventInstanceSchema."
            ),
        },
        {
            "role": "user",
            "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---",
        },
    ]

    try:
        step5c_result = await run_agent_with_retry(
            agent=event_instance_extractor_agent,
            input_data=step5c_input_list,
            config=step5c_run_config,
        )

        if step5c_result:
            potential_output = getattr(step5c_result, "final_output", None)
            if isinstance(potential_output, EventInstanceSchema):
                instance_data = potential_output
            elif isinstance(potential_output, dict):
                try:
                    instance_data = EventInstanceSchema.model_validate(potential_output)
                except ValidationError as e:
                    logger.warning(
                        f"Step 5c dict output failed EventInstanceSchema validation: {e}"
                    )
            else:
                logger.warning(
                    f"Step 5c final_output was not EventInstanceSchema or dict (type: {type(potential_output)})."
                )

            if instance_data and instance_data.identified_instances:
                if instance_data.primary_domain != primary_domain:
                    instance_data.primary_domain = primary_domain
                if not set(instance_data.analyzed_sub_domains):
                    instance_data.analyzed_sub_domains = [
                        sd.sub_domain for sd in sub_domain_data.identified_sub_domains
                    ]
                logger.info(
                    f"Step 5c Result (Structured Instances):\n{instance_data.model_dump_json(indent=2)}"
                )
                print("\n--- Event Instances Extracted (Structured Output) ---")
                print(instance_data.model_dump_json(indent=2))

                output_content = {
                    "primary_domain": instance_data.primary_domain,
                    "analyzed_sub_domains": instance_data.analyzed_sub_domains,
                    "analyzed_event_types": instance_data.analyzed_event_types,
                    "identified_instances": [
                        item.model_dump() for item in instance_data.identified_instances
                    ],
                    "analysis_summary": instance_data.analysis_summary,
                    "analysis_details": {
                        "source_text_length": len(content),
                        "model_used": EVENT_INSTANCE_MODEL,
                        "agent_name": event_instance_extractor_agent.name,
                        "output_schema": EventInstanceSchema.__name__,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    "trace_information": {
                        "trace_id": overall_trace_id or "N/A",
                        "notes": f"Generated by {event_instance_extractor_agent.name} in Step 5c of workflow.",
                    },
                }
                save_result = direct_save_json_output(
                    EVENT_INSTANCE_OUTPUT_DIR,
                    EVENT_INSTANCE_OUTPUT_FILENAME,
                    output_content,
                    overall_trace_id,
                )
                print(f"  - {save_result}")
                logger.info(f"Result of saving event instance output: {save_result}")
            elif instance_data and not instance_data.identified_instances:
                logger.warning(
                    "Step 5c completed but identified_instances list is empty."
                )
                print("\nStep 5c completed, but no event instances were identified.")
            else:
                logger.error(
                    "Step 5c FAILED: Could not extract valid EventInstanceSchema output."
                )
                print("\nError: Failed to extract event instances in Step 5c.")
                instance_data = None
        else:
            logger.error("Step 5c FAILED: Runner.run did not return a result.")
            print(
                "\nError: Failed to get a result from the event instance extraction step."
            )
            instance_data = None

    except (ValidationError, TypeError) as e:
        logger.exception(
            f"Validation or Type error during Step 5c agent run. Error: {e}",
            extra={"trace_id": overall_trace_id or "N/A"},
        )
        print("\nError: A data validation or type issue occurred during Step 5c.")
        print(f"Error details: {e}")
        instance_data = None
    except Exception as e:
        logger.exception(
            "An unexpected error occurred during Step 5c.",
            extra={"trace_id": overall_trace_id or "N/A"},
        )
        print(f"\nAn unexpected error occurred during Step 5c: {type(e).__name__}: {e}")
        instance_data = None

    return instance_data
