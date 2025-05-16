"""Step 5: Event instance extraction functionality."""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import ValidationError

from agentic_team import RunConfig, RunResult, TResponseInputItem

from ..agents import event_instance_extractor_agent
from ..config import EVENT_INSTANCE_MODEL, EVENT_INSTANCE_OUTPUT_DIR, EVENT_INSTANCE_OUTPUT_FILENAME
from ..schemas import EventInstanceSchema, SubDomainSchema, TopicSchema, EventSchema
from ..utils import direct_save_json_output, run_agent_with_retry

logger = logging.getLogger(__name__)

async def extract_event_instances(
    content: str,
    primary_domain: str,
    sub_domain_data: SubDomainSchema,
    topic_data: TopicSchema,
    event_type_data: EventSchema,
    overall_trace_id: Optional[str] = None,
) -> Optional[EventInstanceSchema]:
    """Extract specific event instances from the text using context."""
    if not (primary_domain and sub_domain_data and topic_data and event_type_data):
        logger.info("Skipping Step 5 (Event Instances) because prerequisites were not met.")
        return None

    logger.info(f"--- Running Step 5: Event Instance Extraction (Agent: {event_instance_extractor_agent.name}) ---")
    print(f"\n--- Running Step 5: Event Instance Extraction using model: {EVENT_INSTANCE_MODEL} ---")

    metadata = {
        "workflow_step": "5_event_instance_extraction",
        "agent_name": event_instance_extractor_agent.name,
        "primary_domain_input": primary_domain,
    }
    run_config = RunConfig(trace_metadata={k: str(v) for k, v in metadata.items()})

    context_summary = (
        f"Primary Domain: {primary_domain}\n"
        f"Identified Sub-Domains: {', '.join(sd.sub_domain for sd in sub_domain_data.identified_sub_domains)}\n"
        f"Known Event Types: {', '.join(ev.event_type for ev in event_type_data.identified_events)}"
    )

    input_list: List[TResponseInputItem] = [
        {
            "role": "user",
            "content": (
                "Extract concrete event mentions from the following text. "
                f"Use this context:\n{context_summary}\n\n"
                "Provide the event type, a short text snippet, and a relevance score for each mention. "
                "Output ONLY using the EventInstanceSchema."
            ),
        },
        {"role": "user", "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---"},
    ]

    try:
        result: RunResult = await run_agent_with_retry(
            agent=event_instance_extractor_agent,
            input_data=input_list,
            config=run_config,
        )

        potential_output = getattr(result, "final_output", None)
        instance_data: Optional[EventInstanceSchema] = None
        if isinstance(potential_output, EventInstanceSchema):
            instance_data = potential_output
        elif isinstance(potential_output, dict):
            try:
                instance_data = EventInstanceSchema.model_validate(potential_output)
            except ValidationError as e:
                logger.warning(f"Step 5 output validation failed: {e}")
        if instance_data:
            logger.info("Step 5 Event Instances extracted successfully.")
            save_content = {
                "primary_domain": instance_data.primary_domain,
                "analyzed_sub_domains": instance_data.analyzed_sub_domains,
                "event_instances": [item.model_dump() for item in instance_data.event_instances],
                "analysis_summary": instance_data.analysis_summary,
                "analysis_details": {
                    "source_text_length": len(content),
                    "model_used": EVENT_INSTANCE_MODEL,
                    "agent_name": event_instance_extractor_agent.name,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
                "trace_information": {"trace_id": overall_trace_id or "N/A"},
            }
            result_path = direct_save_json_output(
                EVENT_INSTANCE_OUTPUT_DIR,
                EVENT_INSTANCE_OUTPUT_FILENAME,
                save_content,
                overall_trace_id,
            )
            print(f"  - {result_path}")
        else:
            logger.error("Step 5 FAILED: Could not parse EventInstanceSchema output.")
        return instance_data

    except Exception as e:
        logger.exception("An unexpected error occurred during Step 5 (Event Instances).", exc_info=e)
        print(f"\nAn unexpected error occurred during Step 5: {type(e).__name__}: {e}")
        return None
