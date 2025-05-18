"""Step 4g: Modality type identification functionality."""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import ValidationError

# NOTE: Using the external ``agents`` SDK
try:
    from agents import RunConfig, RunResult, TResponseInputItem  # type: ignore[attr-defined]
except ImportError:
    print("Error: 'agents' SDK library not found or incomplete for step 4g.")
    raise

from ..agents import modality_type_identifier_agent  # Import the new agent
from ..config import (
    MODALITY_TYPE_MODEL,
    MODALITY_TYPE_OUTPUT_DIR,
    MODALITY_TYPE_OUTPUT_FILENAME,
)  # Import new config vars
from ..schemas import (
    ModalityTypeSchema,
    SubDomainSchema,
    TopicSchema,
)  # Import new output schema
from ..utils import direct_save_json_output, run_agent_with_retry

logger = logging.getLogger(__name__)


async def identify_modality_types(
    content: str,
    primary_domain: str,
    sub_domain_data: SubDomainSchema,
    topic_data: TopicSchema,
    overall_trace_id: Optional[str] = None,
) -> Optional[ModalityTypeSchema]:
    """Identify modality types based on domain, sub-domains, and topics.

    Args:
        content: The text content to analyze
        primary_domain: The primary domain identified in step 1
        sub_domain_data: The SubDomainSchema from step 2
        topic_data: The TopicSchema from step 3
        overall_trace_id: The trace ID for logging purposes

    Returns:
        A ModalityTypeSchema object if successful, None otherwise
    """
    if not primary_domain or not sub_domain_data or not topic_data:
        logger.info("Skipping Step 4g because prerequisites were not identified.")
        if not primary_domain:
            print("Skipping Step 4g as primary domain was not identified.")
        elif not sub_domain_data:
            print("Skipping Step 4g as sub-domain identification failed.")
        elif not topic_data:
            print("Skipping Step 4g as topic identification failed.")
        return None

    logger.info(
        f"--- Running Step 4g: Modality Type ID (Agent: {modality_type_identifier_agent.name}) ---"
    )
    print(
        f"\n--- Running Step 4g: Modality Type ID using model: {MODALITY_TYPE_MODEL} ---"
    )

    step4g_metadata_for_trace = {
        "workflow_step": "4g_modality_type_id",
        "agent_name": "Modality Type ID",
        "actual_agent": str(modality_type_identifier_agent.name),
        "primary_domain_input": primary_domain,
        "sub_domains_analyzed_count": str(len(sub_domain_data.identified_sub_domains)),
        "topics_aggregated_count": str(
            sum(len(t.identified_topics) for t in topic_data.sub_domain_topic_map)
        ),
    }
    step4g_run_config = RunConfig(
        trace_metadata={k: str(v) for k, v in step4g_metadata_for_trace.items()}
    )
    step4g_result: Optional[RunResult] = None
    modality_data: Optional[ModalityTypeSchema] = None

    # Prepare context summary for the prompt
    context_summary_for_prompt = (
        f"Primary Domain: {primary_domain}\n"
        f"Identified Sub-Domains: {', '.join(sd.sub_domain for sd in sub_domain_data.identified_sub_domains)}\n"
        f"Previously identified topics (aggregated): {len(topic_data.sub_domain_topic_map)} sub-domains covered with topics."
        # Optionally add more topic detail here if needed
    )

    step4g_input_list: List[TResponseInputItem] = [
        {
            "role": "user",
            "content": (
                f"Analyze the following text to identify key MODALITY types (e.g., Text, Image, Video, Audio, Table, Chart, Code Snippet, Formula). "
                f"Use the provided context:\n{context_summary_for_prompt}\n\n"
                f"Identify modality types relevant to this overall context. "
                f"Output ONLY using the required ModalityTypeSchema, including the primary_domain and analyzed_sub_domains list in the output."
            ),
        },
        {
            "role": "user",
            "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---",
        },
    ]

    try:
        step4g_result = await run_agent_with_retry(
            agent=modality_type_identifier_agent,
            input_data=step4g_input_list,
            config=step4g_run_config,
        )

        if step4g_result:
            potential_output_step4g = getattr(step4g_result, "final_output", None)
            if isinstance(potential_output_step4g, ModalityTypeSchema):
                modality_data = potential_output_step4g
                logger.info(
                    "Successfully extracted ModalityTypeSchema from step4g_result.final_output."
                )
            elif isinstance(potential_output_step4g, dict):
                try:
                    modality_data = ModalityTypeSchema.model_validate(
                        potential_output_step4g
                    )
                    logger.info(
                        "Successfully validated ModalityTypeSchema from step4g_result.final_output dict."
                    )
                except ValidationError as e:
                    logger.warning(
                        f"Step 4g dict output failed ModalityTypeSchema validation: {e}"
                    )
            else:
                logger.warning(
                    f"Step 4g final_output was not ModalityTypeSchema or dict (type: {type(potential_output_step4g)})."
                )

            if modality_data and modality_data.identified_modalities:
                # Ensure context fields match
                if modality_data.primary_domain != primary_domain:
                    logger.warning(
                        f"Primary domain mismatch in Step 4g output ('{modality_data.primary_domain}'). Overwriting with Step 1's ('{primary_domain}')."
                    )
                    modality_data.primary_domain = primary_domain
                if set(modality_data.analyzed_sub_domains) != set(
                    sd.sub_domain for sd in sub_domain_data.identified_sub_domains
                ):
                    logger.warning(
                        f"Analyzed sub-domains in Step 4g output {modality_data.analyzed_sub_domains} differs from Step 2 input { [sd.sub_domain for sd in sub_domain_data.identified_sub_domains]}. Using Step 4g's list."
                    )

                # Log and print results
                modality_log_items = [
                    item.modality_type for item in modality_data.identified_modalities
                ]
                logger.info(
                    f"Step 4g Result: Identified Modality Types = [{', '.join(modality_log_items)}]"
                )
                logger.info(
                    f"Step 4g Result (Structured Modalities):\n{modality_data.model_dump_json(indent=2)}"
                )
                print(
                    "\n--- Modality Types Identified (Structured Output from Step 4g) ---"
                )
                print(modality_data.model_dump_json(indent=2))

                # Save results
                logger.info("Saving modality type identifier output to file...")
                print("\nSaving modality type output file...")
                modality_type_output_content = {
                    "primary_domain": modality_data.primary_domain,
                    "analyzed_sub_domains": modality_data.analyzed_sub_domains,
                    "identified_modalities": [
                        item.model_dump()
                        for item in modality_data.identified_modalities
                    ],
                    "analysis_summary": modality_data.analysis_summary,
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
                        "model_used": MODALITY_TYPE_MODEL,
                        "agent_name": modality_type_identifier_agent.name,
                        "output_schema": ModalityTypeSchema.__name__,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    "trace_information": {
                        "trace_id": overall_trace_id or "N/A",
                        "notes": f"Generated by {modality_type_identifier_agent.name} in Step 4g of workflow.",
                    },
                }
                save_result_step4g = direct_save_json_output(
                    MODALITY_TYPE_OUTPUT_DIR,
                    MODALITY_TYPE_OUTPUT_FILENAME,
                    modality_type_output_content,
                    overall_trace_id,
                )
                print(f"  - {save_result_step4g}")
                logger.info(
                    f"Result of saving modality type output: {save_result_step4g}"
                )

            elif modality_data and not modality_data.identified_modalities:
                logger.warning(
                    "Step 4g completed but identified_modalities list is empty."
                )
                print(
                    "\nStep 4g completed, but no specific modality types were identified."
                )
                # modality_data = None # Don't necessarily signal failure
            else:  # modality_data is None or validation failed
                logger.error(
                    "Step 4g FAILED: Could not extract valid ModalityTypeSchema output."
                )
                print("\nError: Failed to identify modality types in Step 4g.")
                modality_data = None  # Signal failure if this step is critical

        else:
            logger.error("Step 4g FAILED: Runner.run did not return a result.")
            print(
                "\nError: Failed to get a result from the modality type identification step."
            )
            modality_data = None

    except (ValidationError, TypeError) as e:
        logger.exception(
            f"Validation or Type error during Step 4g agent run. Error: {e}",
            extra={"trace_id": overall_trace_id or "N/A"},
        )
        print("\nError: A data validation or type issue occurred during Step 4g.")
        print(f"Error details: {e}")
        modality_data = None
    except Exception as e:
        logger.exception(
            "An unexpected error occurred during Step 4g.",
            extra={"trace_id": overall_trace_id or "N/A"},
        )
        print(f"\nAn unexpected error occurred during Step 4g: {type(e).__name__}: {e}")
        modality_data = None

    return modality_data
