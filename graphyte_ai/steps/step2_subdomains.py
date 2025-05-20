"""Step 2: Sub-domain identification functionality."""

import logging
from datetime import datetime, timezone
from typing import List, Optional, cast

from pydantic import ValidationError

from agents import RunConfig, RunResult, TResponseInputItem  # type: ignore[attr-defined]

from ..workflow_agents import (
    sub_domain_identifier_agent,
    sub_domain_result_agent,
)
from ..config import SUB_DOMAIN_MODEL, SUB_DOMAIN_OUTPUT_DIR, SUB_DOMAIN_OUTPUT_FILENAME
from ..schemas import SubDomainSchema, SubDomainDetail, SubDomainIdentifierSchema
from ..utils import direct_save_json_output, run_agent_with_retry, score_sub_domains

logger = logging.getLogger(__name__)


async def identify_subdomains(
    content: str,
    primary_domain: str,
    trace_id: Optional[str] = None,
    group_id: Optional[str] = None,
) -> Optional[SubDomainSchema]:
    """Identify the sub-domains from the input content.

    Args:
        content: The text content to analyze
        primary_domain: The primary domain identified in step 1
        trace_id: The trace ID for logging purposes
        group_id: The trace group ID for logging purposes

    Returns:
        A SubDomainSchema object if successful, None otherwise
    """
    if not primary_domain:
        logger.info("Skipping Step 2 because no primary domain was identified.")
        print("Skipping Step 2 as primary domain was not identified.")
        return None

    logger.info(
        f"--- Running Step 2: Sub-Domain ID (Agent: {sub_domain_identifier_agent.name}) ---"
    )
    print(
        f"\n--- Running Step 2: Sub-Domain ID for Primary Domain '{primary_domain}' using model: {SUB_DOMAIN_MODEL} ---"
    )

    step2_metadata_for_trace = {
        "workflow_step": "2_sub_domain_id",
        "agent_name": "Sub-Domain ID",
        "actual_agent": str(sub_domain_identifier_agent.name),
        "primary_domain_input": primary_domain,
    }
    step2_run_config = RunConfig(
        workflow_name="step2_subdomains",
        trace_id=trace_id,
        group_id=group_id,
        trace_metadata={k: str(v) for k, v in step2_metadata_for_trace.items()},
    )
    step2_result: Optional[RunResult] = None
    raw_sub_domain_data: Optional[SubDomainIdentifierSchema] = None
    sub_domain_data: Optional[SubDomainSchema] = None

    step2_input_list: List[TResponseInputItem] = [
        {
            "role": "user",
            "content": f"The primary domain of the following text is '{primary_domain}'. Please identify the specific sub-domains within the text related to this primary domain and provide a brief analysis summary.",
        },
        {
            "role": "user",
            "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---",
        },
    ]

    try:
        step2_result = await run_agent_with_retry(
            agent=sub_domain_identifier_agent,
            input_data=step2_input_list,
            config=step2_run_config,
        )

        if step2_result:
            final_agent_obj = getattr(step2_result, "last_agent", None)
            final_agent_name_step2 = getattr(final_agent_obj, "name", "Unknown")
            logger.info(
                f"Step 2 completed. Final agent: {final_agent_name_step2}. Trace ID: {trace_id or 'N/A'}"
            )
            print(f"\nStep 2 finished. Final agent: {final_agent_name_step2}")

            potential_output_step2 = getattr(step2_result, "final_output", None)
            if isinstance(potential_output_step2, SubDomainIdentifierSchema):
                raw_sub_domain_data = potential_output_step2
                logger.info(
                    "Successfully extracted SubDomainIdentifierSchema from step2_result.final_output."
                )
            elif isinstance(potential_output_step2, dict):
                try:
                    raw_sub_domain_data = SubDomainIdentifierSchema.model_validate(
                        potential_output_step2
                    )
                    logger.info(
                        "Successfully validated SubDomainIdentifierSchema from step2_result.final_output dict."
                    )
                except ValidationError as e:
                    logger.warning(
                        f"Step 2 dict output failed SubDomainIdentifierSchema validation: {e}"
                    )
            else:
                logger.warning(
                    f"Step 2 final_output was not SubDomainIdentifierSchema or dict (type: {type(potential_output_step2)})."
                )

            if raw_sub_domain_data and raw_sub_domain_data.identified_sub_domains:
                sub_domain_data = SubDomainSchema(
                    primary_domain=raw_sub_domain_data.primary_domain,
                    identified_sub_domains=[
                        SubDomainDetail(
                            sub_domain=item.sub_domain,
                            confidence_score=None,
                            relevance_score=None,
                            clarity_score=None,
                        )
                        for item in raw_sub_domain_data.identified_sub_domains
                    ],
                    analysis_summary=raw_sub_domain_data.analysis_summary,
                )
                sub_domain_data = cast(SubDomainSchema, sub_domain_data)
                assert sub_domain_data is not None
                scored_data: SubDomainSchema = sub_domain_data
                if (
                    scored_data.primary_domain
                    and scored_data.primary_domain != primary_domain
                ):
                    logger.warning(
                        f"Primary domain mismatch between Step 1 ('{primary_domain}') and Step 2 output ('{sub_domain_data.primary_domain}'). Using Step 1's."
                    )
                    scored_data.primary_domain = primary_domain  # Ensure consistency
                elif not scored_data.primary_domain:
                    scored_data.primary_domain = (
                        primary_domain  # Ensure primary domain is set
                    )

                scored_data = await score_sub_domains(scored_data, content)

                scored_result = await run_agent_with_retry(
                    sub_domain_result_agent,
                    scored_data.model_dump_json(),
                )

                if scored_result:
                    potential_scored_output = getattr(
                        scored_result, "final_output", None
                    )
                    if isinstance(potential_scored_output, SubDomainSchema):
                        scored_data = potential_scored_output
                    elif isinstance(potential_scored_output, dict):
                        try:
                            scored_data = SubDomainSchema.model_validate(
                                potential_scored_output
                            )
                        except ValidationError as e:
                            logger.warning(
                                "SubDomainSchema validation error after scoring: %s",
                                e,
                            )
                            scored_data = SubDomainSchema.model_validate(
                                scored_data.model_dump()
                            )
                    else:
                        logger.error(
                            "Unexpected sub-domain result output type: %s",
                            type(potential_scored_output),
                        )
                        scored_data = SubDomainSchema.model_validate(
                            scored_data.model_dump()
                        )
                else:
                    scored_data = SubDomainSchema.model_validate(
                        scored_data.model_dump()
                    )

                sub_domain_data = scored_data

                sub_domains_items = sub_domain_data.identified_sub_domains
                sub_domains_list = [
                    item.sub_domain.strip()
                    for item in sub_domains_items
                    if item.sub_domain and item.sub_domain.strip()
                ]

                log_items = [f"'{item.sub_domain}'" for item in sub_domains_items]
                logger.info(
                    f"Step 2 Result: Identified Sub-Domains = [{', '.join(log_items)}]"
                )
                logger.info(
                    f"Step 2 Result (Structured Sub-Domains):\n{sub_domain_data.model_dump_json(indent=2)}"
                )
                print(
                    "\n--- Sub-Domains Identified (Structured Output from Step 2) ---"
                )
                print(sub_domain_data.model_dump_json(indent=2))

                if not sub_domains_list:
                    logger.warning(
                        "Step 2 identified sub-domains list, but it's empty after filtering/stripping. Skipping subsequent steps."
                    )
                    print(
                        "\nStep 2 completed, but no specific non-empty sub-domain names were identified. Cannot proceed further."
                    )
                    sub_domain_data = None
                else:
                    logger.info("Saving sub-domain identifier output to file...")
                    print("\nSaving sub-domain output file...")
                    sub_domain_output_content = {
                        "primary_domain": sub_domain_data.primary_domain,
                        "identified_sub_domains": [
                            item.model_dump() for item in sub_domains_items
                        ],
                        "analysis_summary": sub_domain_data.analysis_summary,
                        "analysis_details": {
                            "source_text_length": len(content),
                            "primary_domain_analyzed": primary_domain,
                            "model_used": SUB_DOMAIN_MODEL,
                            "agent_name": sub_domain_identifier_agent.name,
                            "output_schema": SubDomainSchema.__name__,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        },
                        "trace_information": {
                            "trace_id": trace_id or "N/A",
                            "notes": f"Generated by {sub_domain_identifier_agent.name} in Step 2 of workflow.",
                        },
                    }
                    save_result_step2 = direct_save_json_output(
                        SUB_DOMAIN_OUTPUT_DIR,
                        SUB_DOMAIN_OUTPUT_FILENAME,
                        sub_domain_output_content,
                        trace_id,
                    )
                    print(f"  - {save_result_step2}")
                    logger.info(
                        f"Result of saving sub-domain output: {save_result_step2}"
                    )
            else:
                # Handle case where sub_domain_data is None or identified_sub_domains is empty
                final_output_raw = getattr(step2_result, "final_output", None)
                if sub_domain_data and not sub_domain_data.identified_sub_domains:
                    logger.warning(
                        "Step 2 completed but identified_sub_domains list is empty. Skipping subsequent steps."
                    )
                    print(
                        "\nStep 2 completed, but no specific sub-domains were identified. Cannot proceed further."
                    )
                else:  # sub_domain_data is None or validation failed
                    logger.warning(
                        f"Could not extract structured SubDomainSchema or no sub-domains found. Raw final output: {final_output_raw}. Skipping subsequent steps."
                    )
                    print(
                        "\nStep 2 completed, but could not extract structured sub-domain data or none found. Cannot proceed further."
                    )
                sub_domain_data = None

        else:
            logger.error(
                "Step 2 FAILED: Runner.run did not return a result. Skipping subsequent steps."
            )
            print(
                "\nError: Failed to get a result from the sub-domain identification step. Cannot proceed further."
            )
            sub_domain_data = None

    except (ValidationError, TypeError) as e:
        logger.exception(
            f"Validation or Type error during Step 2 agent run. Error: {e}",
            extra={"trace_id": trace_id or "N/A"},
        )
        print("\nError: A data validation or type issue occurred during Step 2.")
        print(f"Error details: {e}")
        sub_domain_data = None
    except Exception as e:
        logger.exception(
            "An unexpected error occurred during Step 2.",
            extra={"trace_id": trace_id or "N/A"},
        )
        print(f"\nAn unexpected error occurred during Step 2: {type(e).__name__}: {e}")
        sub_domain_data = None

    return sub_domain_data
