"""Step 5a: Modality instance extraction functionality."""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import ValidationError

try:
    from agentic_team import RunConfig, RunResult, TResponseInputItem
except ImportError:
    print("Error: 'agentic_team' SDK library not found or incomplete for step 5a.")
    raise

from ..agents import modality_instance_extractor_agent
from ..config import (
    MODALITY_INSTANCE_MODEL,
    MODALITY_INSTANCE_OUTPUT_DIR,
    MODALITY_INSTANCE_OUTPUT_FILENAME,
)
from ..schemas import ModalityInstanceSchema, SubDomainSchema, TopicSchema, ModalityTypeSchema
from ..utils import direct_save_json_output, run_agent_with_retry

logger = logging.getLogger(__name__)

async def identify_modality_instances(
    content: str,
    primary_domain: str,
    sub_domain_data: SubDomainSchema,
    topic_data: TopicSchema,
    modality_type_data: ModalityTypeSchema,
    overall_trace_id: Optional[str] = None,
) -> Optional[ModalityInstanceSchema]:
    """Identify modality occurrences within the text based on previously identified types."""
    if not primary_domain or not sub_domain_data or not topic_data or not modality_type_data:
        logger.info("Skipping Step 5a because prerequisites were not identified.")
        return None

    logger.info(
        f"--- Running Step 5a: Modality Instance Extraction (Agent: {modality_instance_extractor_agent.name}) ---"
    )
    print(f"\n--- Running Step 5a: Modality Instance Extraction using model: {MODALITY_INSTANCE_MODEL} ---")

    trace_metadata = {
        "workflow_step": "5a_modality_instance_extraction",
        "agent_name": "Modality Instance Extraction",
        "actual_agent": str(modality_instance_extractor_agent.name),
        "primary_domain_input": primary_domain,
        "sub_domains_analyzed_count": str(len(sub_domain_data.identified_sub_domains)),
        "modality_types_count": str(len(modality_type_data.identified_modalities)),
    }
    run_config = RunConfig(trace_metadata={k: str(v) for k, v in trace_metadata.items()})

    context_summary = (
        f"Primary Domain: {primary_domain}\n"
        f"Identified Sub-Domains: {', '.join(sd.sub_domain for sd in sub_domain_data.identified_sub_domains)}\n"
        f"Identified Modality Types: {', '.join(m.modality_type for m in modality_type_data.identified_modalities)}"
    )

    step_input: List[TResponseInputItem] = [
        {
            "role": "user",
            "content": (
                "Analyze the following text and list occurrences of each modality type. "
                f"Use this context:\n{context_summary}\n\n"
                "For every occurrence, provide the modality_type and a short snippet. "
                "Output ONLY using the ModalityInstanceSchema."
            ),
        },
        {"role": "user", "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---"},
    ]

    modality_instance_data: Optional[ModalityInstanceSchema] = None
    try:
        result: RunResult = await run_agent_with_retry(
            agent=modality_instance_extractor_agent,
            input_data=step_input,
            config=run_config,
        )
        if result:
            output = getattr(result, "final_output", None)
            if isinstance(output, ModalityInstanceSchema):
                modality_instance_data = output
            elif isinstance(output, dict):
                modality_instance_data = ModalityInstanceSchema.model_validate(output)
            else:
                logger.warning(
                    f"Step 5a final_output was not ModalityInstanceSchema or dict (type: {type(output)})."
                )
        else:
            logger.error("Step 5a FAILED: Runner.run did not return a result.")
    except (ValidationError, TypeError) as e:
        logger.exception(
            f"Validation or Type error during Step 5a agent run. Error: {e}",
            extra={"trace_id": overall_trace_id or 'N/A'},
        )
        modality_instance_data = None
    except Exception as e:
        logger.exception(
            "An unexpected error occurred during Step 5a.", extra={"trace_id": overall_trace_id or 'N/A'}
        )
        modality_instance_data = None

    if modality_instance_data and modality_instance_data.modality_occurrences:
        print("\n--- Modality Instances Identified ---")
        print(modality_instance_data.model_dump_json(indent=2))
        output_content = {
            "primary_domain": modality_instance_data.primary_domain,
            "analyzed_sub_domains": modality_instance_data.analyzed_sub_domains,
            "modality_occurrences": [m.model_dump() for m in modality_instance_data.modality_occurrences],
            "analysis_summary": modality_instance_data.analysis_summary,
            "analysis_details": {
                "source_text_length": len(content),
                "primary_domain_context": primary_domain,
                "sub_domain_context_count": len(sub_domain_data.identified_sub_domains),
                "modality_types_count": len(modality_type_data.identified_modalities),
                "model_used": MODALITY_INSTANCE_MODEL,
                "agent_name": modality_instance_extractor_agent.name,
                "output_schema": ModalityInstanceSchema.__name__,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            "trace_information": {"trace_id": overall_trace_id or "N/A"},
        }
        save_result = direct_save_json_output(
            MODALITY_INSTANCE_OUTPUT_DIR,
            MODALITY_INSTANCE_OUTPUT_FILENAME,
            output_content,
            overall_trace_id,
        )
        print(f"  - {save_result}")
        logger.info(f"Result of saving modality instance output: {save_result}")
    elif modality_instance_data is None:
        print("\nError: Failed to identify modality instances in Step 5a.")
    else:
        logger.warning("Step 5a completed but modality_occurrences list is empty.")
        print("\nStep 5a completed, but no modality occurrences were identified.")

    return modality_instance_data
