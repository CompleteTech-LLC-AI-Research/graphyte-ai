"""Step 1: Domain identification functionality."""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from pydantic import ValidationError

from agents import RunConfig, RunResult  # type: ignore[attr-defined]

from ..workflow_agents import domain_identifier_agent, domain_result_agent
from ..config import DOMAIN_MODEL, DOMAIN_OUTPUT_DIR, DOMAIN_OUTPUT_FILENAME
from ..schemas import DomainSchema, DomainResultSchema
from ..utils import (
    direct_save_json_output,
    run_agent_with_retry,
    run_parallel_scoring,
)

logger = logging.getLogger(__name__)


async def identify_domain(
    content: str,
    trace_id: Optional[str] = None,
    group_id: Optional[str] = None,
    save_output: bool = True,
) -> Optional[DomainResultSchema]:
    """Identify the primary domain from the input content.

    Args:
        content: The text content to analyze
        trace_id: The trace ID for logging purposes
        group_id: The trace group ID for logging purposes
        save_output: Whether to save the scored domain result to file

    Returns:
        A DomainResultSchema object if successful, None otherwise
    """
    logger.info(
        f"--- Running Step 1: Domain ID (Agent: {domain_identifier_agent.name}) ---"
    )
    print(f"--- Running Step 1: Domain ID using model: {DOMAIN_MODEL} ---")

    step1_metadata_for_trace = {
        "workflow_step": "1_domain_id",
        "agent_name": "Domain ID",
        "actual_agent": str(domain_identifier_agent.name),
    }
    step1_run_config = RunConfig(
        workflow_name="step1_domain",
        trace_id=trace_id,
        group_id=group_id,
        trace_metadata={k: str(v) for k, v in step1_metadata_for_trace.items()},
    )
    step1_result: Optional[RunResult] = None
    raw_domain: Optional[DomainSchema] = None
    domain_data: Optional[DomainResultSchema] = None

    try:
        step1_result = await run_agent_with_retry(
            agent=domain_identifier_agent, input_data=content, config=step1_run_config
        )

        if step1_result:
            potential_output = getattr(step1_result, "final_output", None)
            if isinstance(potential_output, DomainSchema):
                raw_domain = potential_output
                logger.info(
                    "Successfully extracted DomainSchema from step1_result.final_output."
                )
            elif isinstance(potential_output, dict):
                try:
                    raw_domain = DomainSchema.model_validate(potential_output)
                    logger.info(
                        "Successfully validated DomainSchema from step1_result.final_output dict."
                    )
                except ValidationError as e:
                    logger.warning(
                        f"Step 1 dict output failed DomainSchema validation: {e}"
                    )
            else:
                logger.warning(
                    f"Step 1 final_output was not DomainSchema or dict (type: {type(potential_output)})."
                )

        if raw_domain and raw_domain.domain:
            primary_domain = raw_domain.domain.strip()
            if primary_domain:
                logger.info(
                    f"Step 1 Result: Primary Domain Identified = {primary_domain}"
                )
                print(f"Step 1 Result: Primary Domain Identified = {primary_domain}")

                conf_data, rel_data, clar_data = await run_parallel_scoring(
                    primary_domain, content
                )

                payload = {
                    "domain": primary_domain,
                    "confidence_score": (
                        conf_data.confidence_score if conf_data else None
                    ),
                    "relevance_score": rel_data.relevance_score if rel_data else None,
                    "clarity_score": clar_data.clarity_score if clar_data else None,
                }

                scored_result = await run_agent_with_retry(
                    domain_result_agent, json.dumps(payload)
                )

                if scored_result:
                    potential_output = getattr(scored_result, "final_output", None)
                    if isinstance(potential_output, DomainResultSchema):
                        domain_data = potential_output
                    elif isinstance(potential_output, dict):
                        try:
                            domain_data = DomainResultSchema.model_validate(
                                potential_output
                            )
                        except ValidationError as e:
                            logger.warning("DomainResultSchema validation error: %s", e)
                            domain_data = DomainResultSchema.model_validate(payload)
                    else:
                        logger.error(
                            "Unexpected domain result output type: %s",
                            type(potential_output),
                        )
                        domain_data = DomainResultSchema.model_validate(payload)
                else:
                    domain_data = DomainResultSchema.model_validate(payload)

                assert domain_data is not None

                logger.info(
                    "Parallel scoring results - confidence: %s, relevance: %s, clarity: %s",
                    domain_data.confidence_score,
                    domain_data.relevance_score,
                    domain_data.clarity_score,
                )

                if save_output:
                    logger.info("Saving scored domain output to file...")
                    print("\nSaving scored domain output file...")
                    domain_output_content: Dict[str, Any] = domain_data.model_dump()
                    domain_output_content["analysis_details"] = {
                        "source_text_length": len(content),
                        "model_used": domain_result_agent.model,
                        "agent_name": domain_result_agent.name,
                        "output_schema": DomainResultSchema.__name__,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    domain_output_content["trace_information"] = {
                        "trace_id": trace_id or "N/A",
                        "notes": f"Generated by {domain_result_agent.name} after scoring in Step 1.",
                    }
                    save_result = direct_save_json_output(
                        DOMAIN_OUTPUT_DIR,
                        DOMAIN_OUTPUT_FILENAME,
                        domain_output_content,
                        trace_id,
                    )
                    print(f"  - {save_result}")
                    logger.info(
                        "Result of saving scored domain output: %s", save_result
                    )
            else:
                logger.error(
                    "Step 1 FAILED: Identified primary domain was empty after stripping. Skipping subsequent steps."
                )
                print(
                    "\nError: Failed to identify a non-empty primary domain in Step 1. Cannot proceed."
                )
                domain_data = None
        else:
            logger.error(
                "Step 1 FAILED: Could not extract valid DomainSchema output. Skipping subsequent steps."
            )
            print(
                "\nError: Failed to identify the primary domain in Step 1. Cannot proceed."
            )
            domain_data = None

    except (ValidationError, TypeError) as e:
        logger.exception(
            f"Validation or Type error during Step 1 agent run. Error: {e}",
            extra={"trace_id": trace_id or "N/A"},
        )
        print("\nError: A data validation or type issue occurred during Step 1.")
        print(f"Error details: {e}")
        domain_data = None
    except Exception as e:
        # Includes potential retry failures from run_agent_with_retry
        logger.exception(
            "An unexpected error occurred during Step 1.",
            extra={"trace_id": trace_id or "N/A"},
        )
        print(f"\nAn unexpected error occurred during Step 1: {type(e).__name__}: {e}")
        domain_data = None

    return domain_data
