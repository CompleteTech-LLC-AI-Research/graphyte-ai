"""Step 4c: Event type identification functionality."""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import ValidationError

from agentic_team import RunConfig, RunResult, TResponseInputItem

from ..agents import event_type_identifier_agent  # Import the new agent
from ..config import (
    EVENT_TYPE_MODEL,
    EVENT_TYPE_OUTPUT_DIR,
    EVENT_TYPE_OUTPUT_FILENAME,
)  # Import new config vars
from ..schemas import (
    EventTypeSchema,
    SubDomainSchema,
    TopicSchema,
)  # Import new output schema
from ..utils import direct_save_json_output, run_agent_with_retry

logger = logging.getLogger(__name__)


async def identify_event_types(
    content: str,
    primary_domain: str,
    sub_domain_data: SubDomainSchema,
    topic_data: TopicSchema,
    overall_trace_id: Optional[str] = None,
) -> Optional[EventTypeSchema]:
    """Identify event types based on domain, sub-domains, and topics.

    Args:
        content: The text content to analyze
        primary_domain: The primary domain identified in step 1
        sub_domain_data: The SubDomainSchema from step 2
        topic_data: The TopicSchema from step 3
        overall_trace_id: The trace ID for logging purposes

    Returns:
        An EventTypeSchema object if successful, None otherwise
    """
    if not primary_domain or not sub_domain_data or not topic_data:
        logger.info("Skipping Step 4c because prerequisites were not identified.")
        if not primary_domain:
            print("Skipping Step 4c as primary domain was not identified.")
        elif not sub_domain_data:
            print("Skipping Step 4c as sub-domain identification failed.")
        elif not topic_data:
            print("Skipping Step 4c as topic identification failed.")
        return None

    logger.info(
        f"--- Running Step 4c: Event Type ID (Agent: {event_type_identifier_agent.name}) ---"
    )
    print(f"\n--- Running Step 4c: Event Type ID using model: {EVENT_TYPE_MODEL} ---")

    step4c_metadata_for_trace = {
        "workflow_step": "4c_event_type_id",
        "agent_name": "Event Type ID",
        "actual_agent": str(event_type_identifier_agent.name),
        "primary_domain_input": primary_domain,
        "sub_domains_analyzed_count": str(len(sub_domain_data.identified_sub_domains)),
        "topics_aggregated_count": str(
            sum(len(t.identified_topics) for t in topic_data.sub_domain_topic_map)
        ),
    }
    step4c_run_config = RunConfig(
        trace_metadata={k: str(v) for k, v in step4c_metadata_for_trace.items()}
    )
    step4c_result: Optional[RunResult] = None
    event_data: Optional[EventTypeSchema] = None

    # Prepare context summary for the prompt
    context_summary_for_prompt = (
        f"Primary Domain: {primary_domain}\n"
        f"Identified Sub-Domains: {', '.join(sd.sub_domain for sd in sub_domain_data.identified_sub_domains)}\n"
        f"Previously identified topics (aggregated): {len(topic_data.sub_domain_topic_map)} sub-domains covered with topics."
        # Optionally add more topic detail here if needed
    )

    step4c_input_list: List[TResponseInputItem] = [
        {
            "role": "user",
            "content": (
                f"Analyze the following text to identify key EVENT types (e.g., Meeting, Acquisition, Conference, Product Launch, Election). "
                f"Focus only on event types. Use the provided context:\n{context_summary_for_prompt}\n\n"
                f"Identify event types relevant to this overall context. "
                f"Output ONLY using the required EventTypeSchema, including the primary_domain and analyzed_sub_domains list in the output."
            ),
        },
        {
            "role": "user",
            "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---",
        },
    ]

    try:
        step4c_result = await run_agent_with_retry(
            agent=event_type_identifier_agent,
            input_data=step4c_input_list,
            config=step4c_run_config,
        )

        if step4c_result:
            potential_output_step4c = getattr(step4c_result, "final_output", None)
            if isinstance(potential_output_step4c, EventTypeSchema):
                event_data = potential_output_step4c
                logger.info(
                    "Successfully extracted EventTypeSchema from step4c_result.final_output."
                )
            elif isinstance(potential_output_step4c, dict):
                try:
                    event_data = EventTypeSchema.model_validate(potential_output_step4c)
                    logger.info(
                        "Successfully validated EventTypeSchema from step4c_result.final_output dict."
                    )
                except ValidationError as e:
                    logger.warning(
                        f"Step 4c dict output failed EventTypeSchema validation: {e}"
                    )
            else:
                logger.warning(
                    f"Step 4c final_output was not EventTypeSchema or dict (type: {type(potential_output_step4c)})."
                )

            if event_data and event_data.identified_events:
                # Ensure context fields match
                if event_data.primary_domain != primary_domain:
                    logger.warning(
                        f"Primary domain mismatch in Step 4c output ('{event_data.primary_domain}'). Overwriting with Step 1's ('{primary_domain}')."
                    )
                    event_data.primary_domain = primary_domain
                if set(event_data.analyzed_sub_domains) != set(
                    sd.sub_domain for sd in sub_domain_data.identified_sub_domains
                ):
                    logger.warning(
                        f"Analyzed sub-domains in Step 4c output {event_data.analyzed_sub_domains} differs from Step 2 input { [sd.sub_domain for sd in sub_domain_data.identified_sub_domains]}. Using Step 4c's list."
                    )

                # Log and print results
                event_log_items = [
                    item.event_type for item in event_data.identified_events
                ]
                logger.info(
                    f"Step 4c Result: Identified Event Types = [{', '.join(event_log_items)}]"
                )
                logger.info(
                    f"Step 4c Result (Structured Events):\n{event_data.model_dump_json(indent=2)}"
                )
                print(
                    "\n--- Event Types Identified (Structured Output from Step 4c) ---"
                )
                print(event_data.model_dump_json(indent=2))

                # Save results
                logger.info("Saving event type identifier output to file...")
                print("\nSaving event type output file...")
                event_type_output_content = {
                    "primary_domain": event_data.primary_domain,
                    "analyzed_sub_domains": event_data.analyzed_sub_domains,
                    "identified_events": [
                        item.model_dump() for item in event_data.identified_events
                    ],
                    "analysis_summary": event_data.analysis_summary,
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
                        "model_used": EVENT_TYPE_MODEL,
                        "agent_name": event_type_identifier_agent.name,
                        "output_schema": EventTypeSchema.__name__,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    "trace_information": {
                        "trace_id": overall_trace_id or "N/A",
                        "notes": f"Generated by {event_type_identifier_agent.name} in Step 4c of workflow.",
                    },
                }
                save_result_step4c = direct_save_json_output(
                    EVENT_TYPE_OUTPUT_DIR,
                    EVENT_TYPE_OUTPUT_FILENAME,
                    event_type_output_content,
                    overall_trace_id,
                )
                print(f"  - {save_result_step4c}")
                logger.info(f"Result of saving event type output: {save_result_step4c}")

            elif event_data and not event_data.identified_events:
                logger.warning("Step 4c completed but identified_events list is empty.")
                print(
                    "\nStep 4c completed, but no specific event types were identified."
                )
                # Don't signal failure for subsequent steps unless necessary
                # event_data = None
            else:  # event_data is None or validation failed
                logger.error(
                    "Step 4c FAILED: Could not extract valid EventTypeSchema output."
                )
                print("\nError: Failed to identify event types in Step 4c.")
                event_data = None  # Signal failure if needed

        else:
            logger.error("Step 4c FAILED: Runner.run did not return a result.")
            print(
                "\nError: Failed to get a result from the event type identification step."
            )
            event_data = None

    except (ValidationError, TypeError) as e:
        logger.exception(
            f"Validation or Type error during Step 4c agent run. Error: {e}",
            extra={"trace_id": overall_trace_id or "N/A"},
        )
        print("\nError: A data validation or type issue occurred during Step 4c.")
        print(f"Error details: {e}")
        event_data = None
    except Exception as e:
        logger.exception(
            "An unexpected error occurred during Step 4c.",
            extra={"trace_id": overall_trace_id or "N/A"},
        )
        print(f"\nAn unexpected error occurred during Step 4c: {type(e).__name__}: {e}")
        event_data = None

    return event_data
