# File: /Users/completetech/Desktop/python-agent-sdk/src/agentic_team_workflow/steps/step4e_evidence_types.py
"""Step 4e: Evidence type identification functionality."""

import logging
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from pydantic import ValidationError

# NOTE: Assuming 'agentic_team' is the correct SDK import alias
try:
    from agentic_team import RunConfig, RunResult, TResponseInputItem
except ImportError:
    print("Error: 'agentic_team' SDK library not found or incomplete for step 4e.")
    raise

from ..agents import evidence_type_identifier_agent # Import the new agent
from ..config import EVIDENCE_TYPE_MODEL, EVIDENCE_TYPE_OUTPUT_DIR, EVIDENCE_TYPE_OUTPUT_FILENAME # Import new config vars
from ..schemas import EvidenceTypeSchema, SubDomainSchema, TopicSchema # Import new output schema
from ..utils import direct_save_json_output, run_agent_with_retry

logger = logging.getLogger(__name__)

async def identify_evidence_types(
    content: str,
    primary_domain: str,
    sub_domain_data: SubDomainSchema,
    topic_data: TopicSchema,
    overall_trace_id: Optional[str] = None
) -> Optional[EvidenceTypeSchema]:
    """Identify evidence types based on domain, sub-domains, and topics.

    Args:
        content: The text content to analyze
        primary_domain: The primary domain identified in step 1
        sub_domain_data: The SubDomainSchema from step 2
        topic_data: The TopicSchema from step 3
        overall_trace_id: The trace ID for logging purposes

    Returns:
        An EvidenceTypeSchema object if successful, None otherwise
    """
    if not primary_domain or not sub_domain_data or not topic_data:
        logger.info("Skipping Step 4e because prerequisites were not identified.")
        if not primary_domain:
            print("Skipping Step 4e as primary domain was not identified.")
        elif not sub_domain_data:
            print("Skipping Step 4e as sub-domain identification failed.")
        elif not topic_data:
            print("Skipping Step 4e as topic identification failed.")
        return None

    logger.info(f"--- Running Step 4e: Evidence Type ID (Agent: {evidence_type_identifier_agent.name}) ---")
    print(f"\n--- Running Step 4e: Evidence Type ID using model: {EVIDENCE_TYPE_MODEL} ---")

    step4e_metadata_for_trace = {
        "workflow_step": "4e_evidence_type_id",
        "agent_name": "Evidence Type ID",
        "actual_agent": str(evidence_type_identifier_agent.name),
        "primary_domain_input": primary_domain,
        "sub_domains_analyzed_count": str(len(sub_domain_data.identified_sub_domains)),
        "topics_aggregated_count": str(sum(len(t.identified_topics) for t in topic_data.sub_domain_topic_map)),
    }
    step4e_run_config = RunConfig(trace_metadata={k: str(v) for k, v in step4e_metadata_for_trace.items()})
    step4e_result: Optional[RunResult] = None
    evidence_data: Optional[EvidenceTypeSchema] = None

    # Prepare context summary for the prompt
    context_summary_for_prompt = (
        f"Primary Domain: {primary_domain}\n"
        f"Identified Sub-Domains: {', '.join(sd.sub_domain for sd in sub_domain_data.identified_sub_domains)}\n"
        f"Previously identified topics (aggregated): {len(topic_data.sub_domain_topic_map)} sub-domains covered with topics."
        # Optionally add more topic detail here if needed
    )

    step4e_input_list: List[TResponseInputItem] = [
        {"role": "user", "content": (
            f"Analyze the following text to identify key EVIDENCE types (e.g., Testimony, Document, Statistic, Anecdote, Expert Opinion). "
            f"Use the provided context:\n{context_summary_for_prompt}\n\n"
            f"Identify evidence types relevant to this overall context. "
            f"Output ONLY using the required EvidenceTypeSchema, including the primary_domain and analyzed_sub_domains list in the output."
        )},
        {"role": "user", "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---"}
    ]

    try:
        step4e_result = await run_agent_with_retry(
            agent=evidence_type_identifier_agent,
            input_data=step4e_input_list,
            config=step4e_run_config
        )

        if step4e_result:
            potential_output_step4e = getattr(step4e_result, 'final_output', None)
            if isinstance(potential_output_step4e, EvidenceTypeSchema):
                evidence_data = potential_output_step4e
                logger.info("Successfully extracted EvidenceTypeSchema from step4e_result.final_output.")
            elif isinstance(potential_output_step4e, dict):
                try:
                    evidence_data = EvidenceTypeSchema.model_validate(potential_output_step4e)
                    logger.info("Successfully validated EvidenceTypeSchema from step4e_result.final_output dict.")
                except ValidationError as e:
                    logger.warning(f"Step 4e dict output failed EvidenceTypeSchema validation: {e}")
            else:
                logger.warning(f"Step 4e final_output was not EvidenceTypeSchema or dict (type: {type(potential_output_step4e)}).")

            if evidence_data and evidence_data.identified_evidence:
                # Ensure context fields match
                if evidence_data.primary_domain != primary_domain:
                    logger.warning(f"Primary domain mismatch in Step 4e output ('{evidence_data.primary_domain}'). Overwriting with Step 1's ('{primary_domain}').")
                    evidence_data.primary_domain = primary_domain
                if set(evidence_data.analyzed_sub_domains) != set(sd.sub_domain for sd in sub_domain_data.identified_sub_domains):
                    logger.warning(f"Analyzed sub-domains in Step 4e output {evidence_data.analyzed_sub_domains} differs from Step 2 input { [sd.sub_domain for sd in sub_domain_data.identified_sub_domains]}. Using Step 4e's list.")

                # Log and print results
                evidence_log_items = [item.evidence_type for item in evidence_data.identified_evidence]
                logger.info(f"Step 4e Result: Identified Evidence Types = [{', '.join(evidence_log_items)}]")
                logger.info(f"Step 4e Result (Structured Evidence):\n{evidence_data.model_dump_json(indent=2)}")
                print("\n--- Evidence Types Identified (Structured Output from Step 4e) ---")
                print(evidence_data.model_dump_json(indent=2))

                # Save results
                logger.info("Saving evidence type identifier output to file...")
                print("\nSaving evidence type output file...")
                evidence_type_output_content = {
                    "primary_domain": evidence_data.primary_domain,
                    "analyzed_sub_domains": evidence_data.analyzed_sub_domains,
                    "identified_evidence": [item.model_dump() for item in evidence_data.identified_evidence],
                    "analysis_summary": evidence_data.analysis_summary,
                    "analysis_details": {
                        "source_text_length": len(content),
                        "primary_domain_context": primary_domain,
                        "sub_domain_context_count": len(sub_domain_data.identified_sub_domains),
                        "topic_context_count": sum(len(t.identified_topics) for t in topic_data.sub_domain_topic_map),
                        "model_used": EVIDENCE_TYPE_MODEL,
                        "agent_name": evidence_type_identifier_agent.name,
                        "output_schema": EvidenceTypeSchema.__name__,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat()
                    },
                    "trace_information": {
                        "trace_id": overall_trace_id or "N/A",
                        "notes": f"Generated by {evidence_type_identifier_agent.name} in Step 4e of workflow."
                    }
                }
                save_result_step4e = direct_save_json_output(
                    EVIDENCE_TYPE_OUTPUT_DIR, EVIDENCE_TYPE_OUTPUT_FILENAME, evidence_type_output_content, overall_trace_id
                )
                print(f"  - {save_result_step4e}")
                logger.info(f"Result of saving evidence type output: {save_result_step4e}")

            elif evidence_data and not evidence_data.identified_evidence:
                logger.warning("Step 4e completed but identified_evidence list is empty.")
                print("\nStep 4e completed, but no specific evidence types were identified.")
                # Don't necessarily signal failure for the whole workflow if this step finds nothing
                # evidence_data = None
            else: # evidence_data is None or validation failed
                logger.error("Step 4e FAILED: Could not extract valid EvidenceTypeSchema output.")
                print("\nError: Failed to identify evidence types in Step 4e.")
                evidence_data = None # Signal failure if this step is critical

        else:
            logger.error("Step 4e FAILED: Runner.run did not return a result.")
            print("\nError: Failed to get a result from the evidence type identification step.")
            evidence_data = None

    except (ValidationError, TypeError) as e:
        logger.exception(f"Validation or Type error during Step 4e agent run. Error: {e}", extra={"trace_id": overall_trace_id or 'N/A'})
        print(f"\nError: A data validation or type issue occurred during Step 4e.")
        print(f"Error details: {e}")
        evidence_data = None
    except Exception as e:
        logger.exception("An unexpected error occurred during Step 4e.", extra={"trace_id": overall_trace_id or 'N/A'})
        print(f"\nAn unexpected error occurred during Step 4e: {type(e).__name__}: {e}")
        evidence_data = None

    return evidence_data