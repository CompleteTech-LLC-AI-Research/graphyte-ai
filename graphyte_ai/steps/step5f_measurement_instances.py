"""Step 5f: Measurement instance extraction functionality."""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import ValidationError

from agents import RunConfig, RunResult, TResponseInputItem  # type: ignore[attr-defined]

from ..workflow_agents import measurement_instance_extractor_agent
from ..config import (
    MEASUREMENT_INSTANCE_MODEL,
    MEASUREMENT_INSTANCE_OUTPUT_DIR,
    MEASUREMENT_INSTANCE_OUTPUT_FILENAME,
)
from ..schemas import (
    MeasurementInstanceSchema,
    SubDomainSchema,
    TopicSchema,
    MeasurementTypeSchema,
)
from ..utils import (
    direct_save_json_output,
    run_agent_with_retry,
    score_measurement_instances,
)

logger = logging.getLogger(__name__)


async def identify_measurement_instances(
    content: str,
    primary_domain: str,
    sub_domain_data: SubDomainSchema,
    topic_data: TopicSchema,
    measurement_data: MeasurementTypeSchema,
    trace_id: Optional[str] = None,
    group_id: Optional[str] = None,
) -> Optional[MeasurementInstanceSchema]:
    """Extract measurement mentions from the text based on context."""
    if (
        not primary_domain
        or not sub_domain_data
        or not topic_data
        or not measurement_data
    ):
        logger.info("Skipping Step 5f because prerequisites were not identified.")
        if not primary_domain:
            print("Skipping Step 5f as primary domain was not identified.")
        elif not sub_domain_data:
            print("Skipping Step 5f as sub-domain identification failed.")
        elif not topic_data:
            print("Skipping Step 5f as topic identification failed.")
        elif not measurement_data:
            print("Skipping Step 5f as measurement type identification failed.")
        return None

    logger.info(
        f"--- Running Step 5f: Measurement Instance Extraction (Agent: {measurement_instance_extractor_agent.name}) ---"
    )
    print(
        f"\n--- Running Step 5f: Measurement Instance Extraction using model: {MEASUREMENT_INSTANCE_MODEL} ---"
    )

    step5f_metadata_for_trace = {
        "workflow_step": "5f_measurement_instance_extraction",
        "agent_name": "Measurement Instance Extractor",
        "actual_agent": str(measurement_instance_extractor_agent.name),
        "primary_domain_input": primary_domain,
        "sub_domains_analyzed_count": str(len(sub_domain_data.identified_sub_domains)),
        "topic_context_count": str(
            sum(len(t.identified_topics) for t in topic_data.sub_domain_topic_map)
        ),
        "measurement_type_count": str(len(measurement_data.identified_measurements)),
    }
    step5f_run_config = RunConfig(
        workflow_name="step5f_measurement_instances",
        trace_id=trace_id,
        group_id=group_id,
        trace_metadata={k: str(v) for k, v in step5f_metadata_for_trace.items()},
    )
    step5f_result: Optional[RunResult] = None
    instance_data: Optional[MeasurementInstanceSchema] = None

    context_summary_for_prompt = (
        f"Primary Domain: {primary_domain}\n"
        f"Identified Sub-Domains: {', '.join(sd.sub_domain for sd in sub_domain_data.identified_sub_domains)}\n"
        f"Measurement Types Considered: {', '.join(m.measurement_type for m in measurement_data.identified_measurements)}"
    )

    step5f_input_list: List[TResponseInputItem] = [
        {
            "role": "user",
            "content": (
                f"Extract specific measurement mentions from the text. Use the provided context:\n{context_summary_for_prompt}\n\n"
                f"Provide the measurement type, exact text span and character offsets. Output ONLY using the required MeasurementInstanceSchema."
            ),
        },
        {
            "role": "user",
            "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---",
        },
    ]

    try:
        step5f_result = await run_agent_with_retry(
            agent=measurement_instance_extractor_agent,
            input_data=step5f_input_list,
            config=step5f_run_config,
        )

        if step5f_result:
            potential_output = getattr(step5f_result, "final_output", None)
            if isinstance(potential_output, MeasurementInstanceSchema):
                instance_data = potential_output
            elif isinstance(potential_output, dict):
                try:
                    instance_data = MeasurementInstanceSchema.model_validate(
                        potential_output
                    )
                except ValidationError as e:
                    logger.warning(
                        f"Step 5f dict output failed MeasurementInstanceSchema validation: {e}"
                    )
            else:
                logger.warning(
                    f"Step 5f final_output was not MeasurementInstanceSchema or dict (type: {type(potential_output)})."
                )

            if instance_data and instance_data.identified_instances:
                if instance_data.primary_domain != primary_domain:
                    instance_data.primary_domain = primary_domain
                if not set(instance_data.analyzed_sub_domains):
                    instance_data.analyzed_sub_domains = [
                        sd.sub_domain for sd in sub_domain_data.identified_sub_domains
                    ]
                instance_data = await score_measurement_instances(
                    instance_data, content
                )
                logger.info(
                    f"Step 5f Result (Structured Instances):\n{instance_data.model_dump_json(indent=2)}"
                )
                print("\n--- Measurement Instances Extracted (Structured Output) ---")
                print(instance_data.model_dump_json(indent=2))

                output_content = {
                    "primary_domain": instance_data.primary_domain,
                    "analyzed_sub_domains": instance_data.analyzed_sub_domains,
                    "analyzed_measurement_types": instance_data.analyzed_measurement_types,
                    "identified_instances": [
                        item.model_dump() for item in instance_data.identified_instances
                    ],
                    "analysis_summary": instance_data.analysis_summary,
                    "analysis_details": {
                        "source_text_length": len(content),
                        "model_used": MEASUREMENT_INSTANCE_MODEL,
                        "agent_name": measurement_instance_extractor_agent.name,
                        "output_schema": MeasurementInstanceSchema.__name__,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    "trace_information": {
                        "trace_id": trace_id or "N/A",
                        "notes": f"Generated by {measurement_instance_extractor_agent.name} in Step 5f of workflow.",
                    },
                }
                save_result = direct_save_json_output(
                    MEASUREMENT_INSTANCE_OUTPUT_DIR,
                    MEASUREMENT_INSTANCE_OUTPUT_FILENAME,
                    output_content,
                    trace_id,
                )
                print(f"  - {save_result}")
                logger.info(
                    f"Result of saving measurement instance output: {save_result}"
                )
            elif instance_data and not instance_data.identified_instances:
                logger.warning(
                    "Step 5f completed but identified_instances list is empty."
                )
                print(
                    "\nStep 5f completed, but no measurement instances were identified."
                )
            else:
                logger.error(
                    "Step 5f FAILED: Could not extract valid MeasurementInstanceSchema output."
                )
                print("\nError: Failed to extract measurement instances in Step 5f.")
                instance_data = None
        else:
            logger.error("Step 5f FAILED: Runner.run did not return a result.")
            print(
                "\nError: Failed to get a result from the measurement instance extraction step."
            )
            instance_data = None

    except (ValidationError, TypeError) as e:
        logger.exception(
            f"Validation or Type error during Step 5f agent run. Error: {e}",
            extra={"trace_id": trace_id or "N/A"},
        )
        print("\nError: A data validation or type issue occurred during Step 5f.")
        print(f"Error details: {e}")
        instance_data = None
    except Exception as e:
        logger.exception(
            "An unexpected error occurred during Step 5f.",
            extra={"trace_id": trace_id or "N/A"},
        )
        print(f"\nAn unexpected error occurred during Step 5f: {type(e).__name__}: {e}")
        instance_data = None

    return instance_data
