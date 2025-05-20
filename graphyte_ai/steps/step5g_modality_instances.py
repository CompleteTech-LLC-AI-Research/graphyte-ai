"""Step 5g: Modality instance extraction functionality."""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import ValidationError

from agents import RunConfig, RunResult, TResponseInputItem  # type: ignore[attr-defined]

from ..workflow_agents import modality_instance_extractor_agent
from ..config import (
    MODALITY_INSTANCE_MODEL,
    MODALITY_INSTANCE_OUTPUT_DIR,
    MODALITY_INSTANCE_OUTPUT_FILENAME,
)
from ..schemas import (
    ModalityInstanceSchema,
    SubDomainSchema,
    TopicSchema,
    ModalityTypeSchema,
)
from ..utils import (
    direct_save_json_output,
    run_agent_with_retry,
    score_modality_instances,
)

logger = logging.getLogger(__name__)


async def identify_modality_instances(
    content: str,
    primary_domain: str,
    sub_domain_data: SubDomainSchema,
    topic_data: TopicSchema,
    modality_data: ModalityTypeSchema,
    trace_id: Optional[str] = None,
    group_id: Optional[str] = None,
) -> Optional[ModalityInstanceSchema]:
    """Extract modality mentions from the text based on context."""
    if not primary_domain or not sub_domain_data or not topic_data or not modality_data:
        logger.info("Skipping Step 5g because prerequisites were not identified.")
        if not primary_domain:
            print("Skipping Step 5g as primary domain was not identified.")
        elif not sub_domain_data:
            print("Skipping Step 5g as sub-domain identification failed.")
        elif not topic_data:
            print("Skipping Step 5g as topic identification failed.")
        elif not modality_data:
            print("Skipping Step 5g as modality type identification failed.")
        return None

    logger.info(
        f"--- Running Step 5g: Modality Instance Extraction (Agent: {modality_instance_extractor_agent.name}) ---"
    )
    print(
        f"\n--- Running Step 5g: Modality Instance Extraction using model: {MODALITY_INSTANCE_MODEL} ---"
    )

    step5g_metadata_for_trace = {
        "workflow_step": "5g_modality_instance_extraction",
        "agent_name": "Modality Instance Extractor",
        "actual_agent": str(modality_instance_extractor_agent.name),
        "primary_domain_input": primary_domain,
        "sub_domains_analyzed_count": str(len(sub_domain_data.identified_sub_domains)),
        "topic_context_count": str(
            sum(len(t.identified_topics) for t in topic_data.sub_domain_topic_map)
        ),
        "modality_type_count": str(len(modality_data.identified_modalities)),
    }
    step5g_run_config = RunConfig(
        workflow_name="step5g_modality_instances",
        trace_id=trace_id,
        group_id=group_id,
        trace_metadata={k: str(v) for k, v in step5g_metadata_for_trace.items()},
    )
    step5g_result: Optional[RunResult] = None
    instance_data: Optional[ModalityInstanceSchema] = None

    context_summary_for_prompt = (
        f"Primary Domain: {primary_domain}\n"
        f"Identified Sub-Domains: {', '.join(sd.sub_domain for sd in sub_domain_data.identified_sub_domains)}\n"
        f"Modality Types Considered: {', '.join(m.modality_type for m in modality_data.identified_modalities)}"
    )

    step5g_input_list: List[TResponseInputItem] = [
        {
            "role": "user",
            "content": (
                f"Extract specific modality mentions from the text. Use the provided context:\n{context_summary_for_prompt}\n\n"
                f"Provide the modality type, exact text span and character offsets. Output ONLY using the required ModalityInstanceSchema."
            ),
        },
        {
            "role": "user",
            "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---",
        },
    ]

    try:
        step5g_result = await run_agent_with_retry(
            agent=modality_instance_extractor_agent,
            input_data=step5g_input_list,
            config=step5g_run_config,
        )

        if step5g_result:
            potential_output = getattr(step5g_result, "final_output", None)
            if isinstance(potential_output, ModalityInstanceSchema):
                instance_data = potential_output
            elif isinstance(potential_output, dict):
                try:
                    instance_data = ModalityInstanceSchema.model_validate(
                        potential_output
                    )
                except ValidationError as e:
                    logger.warning(
                        f"Step 5g dict output failed ModalityInstanceSchema validation: {e}"
                    )
            else:
                logger.warning(
                    f"Step 5g final_output was not ModalityInstanceSchema or dict (type: {type(potential_output)})."
                )

            if instance_data and instance_data.identified_instances:
                if instance_data.primary_domain != primary_domain:
                    instance_data.primary_domain = primary_domain
                if not set(instance_data.analyzed_sub_domains):
                    instance_data.analyzed_sub_domains = [
                        sd.sub_domain for sd in sub_domain_data.identified_sub_domains
                    ]
                instance_data = await score_modality_instances(instance_data, content)
                logger.info(
                    f"Step 5g Result (Structured Instances):\n{instance_data.model_dump_json(indent=2)}"
                )
                print("\n--- Modality Instances Extracted (Structured Output) ---")
                print(instance_data.model_dump_json(indent=2))

                output_content = {
                    "primary_domain": instance_data.primary_domain,
                    "analyzed_sub_domains": instance_data.analyzed_sub_domains,
                    "analyzed_modality_types": instance_data.analyzed_modality_types,
                    "identified_instances": [
                        item.model_dump() for item in instance_data.identified_instances
                    ],
                    "analysis_summary": instance_data.analysis_summary,
                    "analysis_details": {
                        "source_text_length": len(content),
                        "model_used": MODALITY_INSTANCE_MODEL,
                        "agent_name": modality_instance_extractor_agent.name,
                        "output_schema": ModalityInstanceSchema.__name__,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    "trace_information": {
                        "trace_id": trace_id or "N/A",
                        "notes": f"Generated by {modality_instance_extractor_agent.name} in Step 5g of workflow.",
                    },
                }
                save_result = direct_save_json_output(
                    MODALITY_INSTANCE_OUTPUT_DIR,
                    MODALITY_INSTANCE_OUTPUT_FILENAME,
                    output_content,
                    trace_id,
                )
                print(f"  - {save_result}")
                logger.info(f"Result of saving modality instance output: {save_result}")
            elif instance_data and not instance_data.identified_instances:
                logger.warning(
                    "Step 5g completed but identified_instances list is empty."
                )
                print("\nStep 5g completed, but no modality instances were identified.")
            else:
                logger.error(
                    "Step 5g FAILED: Could not extract valid ModalityInstanceSchema output."
                )
                print("\nError: Failed to extract modality instances in Step 5g.")
                instance_data = None
        else:
            logger.error("Step 5g FAILED: Runner.run did not return a result.")
            print(
                "\nError: Failed to get a result from the modality instance extraction step."
            )
            instance_data = None

    except (ValidationError, TypeError) as e:
        logger.exception(
            f"Validation or Type error during Step 5g agent run. Error: {e}",
            extra={"trace_id": trace_id or "N/A"},
        )
        print("\nError: A data validation or type issue occurred during Step 5g.")
        print(f"Error details: {e}")
        instance_data = None
    except Exception as e:
        logger.exception(
            "An unexpected error occurred during Step 5g.",
            extra={"trace_id": trace_id or "N/A"},
        )
        print(f"\nAn unexpected error occurred during Step 5g: {type(e).__name__}: {e}")
        instance_data = None

    return instance_data
