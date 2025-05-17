# File: /Users/completetech/Desktop/python-agent-sdk/src/agentic_team_workflow/steps/step4f_measurement_types.py
"""Step 4f: Measurement type identification functionality."""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import ValidationError

# NOTE: Assuming 'agentic_team' is the correct SDK import alias
try:
    from agentic_team import RunConfig, RunResult, TResponseInputItem
except ImportError:
    print("Error: 'agentic_team' SDK library not found or incomplete for step 4f.")
    raise

from ..agents import measurement_type_identifier_agent  # Import the new agent
from ..config import (
    MEASUREMENT_TYPE_MODEL,
    MEASUREMENT_TYPE_OUTPUT_DIR,
    MEASUREMENT_TYPE_OUTPUT_FILENAME,
)  # Import new config vars
from ..schemas import (
    MeasurementTypeSchema,
    SubDomainSchema,
    TopicSchema,
)  # Import new output schema
from ..utils import direct_save_json_output, run_agent_with_retry

logger = logging.getLogger(__name__)


async def identify_measurement_types(
    content: str,
    primary_domain: str,
    sub_domain_data: SubDomainSchema,
    topic_data: TopicSchema,
    overall_trace_id: Optional[str] = None,
) -> Optional[MeasurementTypeSchema]:
    """Identify measurement types based on domain, sub-domains, and topics.

    Args:
        content: The text content to analyze
        primary_domain: The primary domain identified in step 1
        sub_domain_data: The SubDomainSchema from step 2
        topic_data: The TopicSchema from step 3
        overall_trace_id: The trace ID for logging purposes

    Returns:
        A MeasurementTypeSchema object if successful, None otherwise
    """
    if not primary_domain or not sub_domain_data or not topic_data:
        logger.info("Skipping Step 4f because prerequisites were not identified.")
        if not primary_domain:
            print("Skipping Step 4f as primary domain was not identified.")
        elif not sub_domain_data:
            print("Skipping Step 4f as sub-domain identification failed.")
        elif not topic_data:
            print("Skipping Step 4f as topic identification failed.")
        return None

    logger.info(
        f"--- Running Step 4f: Measurement Type ID (Agent: {measurement_type_identifier_agent.name}) ---"
    )
    print(
        f"\n--- Running Step 4f: Measurement Type ID using model: {MEASUREMENT_TYPE_MODEL} ---"
    )

    step4f_metadata_for_trace = {
        "workflow_step": "4f_measurement_type_id",
        "agent_name": "Measurement Type ID",
        "actual_agent": str(measurement_type_identifier_agent.name),
        "primary_domain_input": primary_domain,
        "sub_domains_analyzed_count": str(len(sub_domain_data.identified_sub_domains)),
        "topics_aggregated_count": str(
            sum(len(t.identified_topics) for t in topic_data.sub_domain_topic_map)
        ),
    }
    step4f_run_config = RunConfig(
        trace_metadata={k: str(v) for k, v in step4f_metadata_for_trace.items()}
    )
    step4f_result: Optional[RunResult] = None
    measurement_data: Optional[MeasurementTypeSchema] = None

    # Prepare context summary for the prompt
    context_summary_for_prompt = (
        f"Primary Domain: {primary_domain}\n"
        f"Identified Sub-Domains: {', '.join(sd.sub_domain for sd in sub_domain_data.identified_sub_domains)}\n"
        f"Previously identified topics (aggregated): {len(topic_data.sub_domain_topic_map)} sub-domains covered with topics."
        # Optionally add more topic detail here if needed
    )

    step4f_input_list: List[TResponseInputItem] = [
        {
            "role": "user",
            "content": (
                f"Analyze the following text to identify key MEASUREMENT types (e.g., Financial Metric, Physical Quantity, Performance Indicator, Survey Result, Count, Ratio, Percentage). "
                f"Use the provided context:\n{context_summary_for_prompt}\n\n"
                f"Identify measurement types relevant to this overall context. "
                f"Output ONLY using the required MeasurementTypeSchema, including the primary_domain and analyzed_sub_domains list in the output."
            ),
        },
        {
            "role": "user",
            "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---",
        },
    ]

    try:
        step4f_result = await run_agent_with_retry(
            agent=measurement_type_identifier_agent,
            input_data=step4f_input_list,
            config=step4f_run_config,
        )

        if step4f_result:
            potential_output_step4f = getattr(step4f_result, "final_output", None)
            if isinstance(potential_output_step4f, MeasurementTypeSchema):
                measurement_data = potential_output_step4f
                logger.info(
                    "Successfully extracted MeasurementTypeSchema from step4f_result.final_output."
                )
            elif isinstance(potential_output_step4f, dict):
                try:
                    measurement_data = MeasurementTypeSchema.model_validate(
                        potential_output_step4f
                    )
                    logger.info(
                        "Successfully validated MeasurementTypeSchema from step4f_result.final_output dict."
                    )
                except ValidationError as e:
                    logger.warning(
                        f"Step 4f dict output failed MeasurementTypeSchema validation: {e}"
                    )
            else:
                logger.warning(
                    f"Step 4f final_output was not MeasurementTypeSchema or dict (type: {type(potential_output_step4f)})."
                )

            if measurement_data and measurement_data.identified_measurements:
                # Ensure context fields match
                if measurement_data.primary_domain != primary_domain:
                    logger.warning(
                        f"Primary domain mismatch in Step 4f output ('{measurement_data.primary_domain}'). Overwriting with Step 1's ('{primary_domain}')."
                    )
                    measurement_data.primary_domain = primary_domain
                if set(measurement_data.analyzed_sub_domains) != set(
                    sd.sub_domain for sd in sub_domain_data.identified_sub_domains
                ):
                    logger.warning(
                        f"Analyzed sub-domains in Step 4f output {measurement_data.analyzed_sub_domains} differs from Step 2 input { [sd.sub_domain for sd in sub_domain_data.identified_sub_domains]}. Using Step 4f's list."
                    )

                # Log and print results
                measurement_log_items = [
                    item.measurement_type
                    for item in measurement_data.identified_measurements
                ]
                logger.info(
                    f"Step 4f Result: Identified Measurement Types = [{', '.join(measurement_log_items)}]"
                )
                logger.info(
                    f"Step 4f Result (Structured Measurements):\n{measurement_data.model_dump_json(indent=2)}"
                )
                print(
                    "\n--- Measurement Types Identified (Structured Output from Step 4f) ---"
                )
                print(measurement_data.model_dump_json(indent=2))

                # Save results
                logger.info("Saving measurement type identifier output to file...")
                print("\nSaving measurement type output file...")
                measurement_type_output_content = {
                    "primary_domain": measurement_data.primary_domain,
                    "analyzed_sub_domains": measurement_data.analyzed_sub_domains,
                    "identified_measurements": [
                        item.model_dump()
                        for item in measurement_data.identified_measurements
                    ],
                    "analysis_summary": measurement_data.analysis_summary,
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
                        "model_used": MEASUREMENT_TYPE_MODEL,
                        "agent_name": measurement_type_identifier_agent.name,
                        "output_schema": MeasurementTypeSchema.__name__,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    "trace_information": {
                        "trace_id": overall_trace_id or "N/A",
                        "notes": f"Generated by {measurement_type_identifier_agent.name} in Step 4f of workflow.",
                    },
                }
                save_result_step4f = direct_save_json_output(
                    MEASUREMENT_TYPE_OUTPUT_DIR,
                    MEASUREMENT_TYPE_OUTPUT_FILENAME,
                    measurement_type_output_content,
                    overall_trace_id,
                )
                print(f"  - {save_result_step4f}")
                logger.info(
                    f"Result of saving measurement type output: {save_result_step4f}"
                )

            elif measurement_data and not measurement_data.identified_measurements:
                logger.warning(
                    "Step 4f completed but identified_measurements list is empty."
                )
                print(
                    "\nStep 4f completed, but no specific measurement types were identified."
                )
                # measurement_data = None # Don't necessarily signal failure
            else:  # measurement_data is None or validation failed
                logger.error(
                    "Step 4f FAILED: Could not extract valid MeasurementTypeSchema output."
                )
                print("\nError: Failed to identify measurement types in Step 4f.")
                measurement_data = None  # Signal failure if this step is critical

        else:
            logger.error("Step 4f FAILED: Runner.run did not return a result.")
            print(
                "\nError: Failed to get a result from the measurement type identification step."
            )
            measurement_data = None

    except (ValidationError, TypeError) as e:
        logger.exception(
            f"Validation or Type error during Step 4f agent run. Error: {e}",
            extra={"trace_id": overall_trace_id or "N/A"},
        )
        print("\nError: A data validation or type issue occurred during Step 4f.")
        print(f"Error details: {e}")
        measurement_data = None
    except Exception as e:
        logger.exception(
            "An unexpected error occurred during Step 4f.",
            extra={"trace_id": overall_trace_id or "N/A"},
        )
        print(f"\nAn unexpected error occurred during Step 4f: {type(e).__name__}: {e}")
        measurement_data = None

    return measurement_data
