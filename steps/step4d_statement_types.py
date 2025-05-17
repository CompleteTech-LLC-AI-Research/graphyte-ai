# File: /Users/completetech/Desktop/python-agent-sdk/src/agentic_team_workflow/steps/step4d_statement_types.py
"""Step 4d: Statement type identification functionality."""

import logging
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from pydantic import ValidationError

# NOTE: Assuming 'agentic_team' is the correct SDK import alias
try:
    from agentic_team import RunConfig, RunResult, TResponseInputItem
except ImportError:
    print("Error: 'agentic_team' SDK library not found or incomplete for step 4d.")
    raise

from ..agents import statement_type_identifier_agent # Import the new agent
from ..config import STATEMENT_TYPE_MODEL, STATEMENT_TYPE_OUTPUT_DIR, STATEMENT_TYPE_OUTPUT_FILENAME # Import new config vars
from ..schemas import StatementTypeSchema, SubDomainSchema, TopicSchema # Import new output schema
from ..utils import direct_save_json_output, run_agent_with_retry

logger = logging.getLogger(__name__)

async def identify_statement_types(
    content: str,
    primary_domain: str,
    sub_domain_data: SubDomainSchema,
    topic_data: TopicSchema,
    overall_trace_id: Optional[str] = None
) -> Optional[StatementTypeSchema]:
    """Identify statement types based on domain, sub-domains, and topics.

    Args:
        content: The text content to analyze
        primary_domain: The primary domain identified in step 1
        sub_domain_data: The SubDomainSchema from step 2
        topic_data: The TopicSchema from step 3
        overall_trace_id: The trace ID for logging purposes

    Returns:
        A StatementTypeSchema object if successful, None otherwise
    """
    if not primary_domain or not sub_domain_data or not topic_data:
        logger.info("Skipping Step 4d because prerequisites were not identified.")
        if not primary_domain:
            print("Skipping Step 4d as primary domain was not identified.")
        elif not sub_domain_data:
            print("Skipping Step 4d as sub-domain identification failed.")
        elif not topic_data:
            print("Skipping Step 4d as topic identification failed.")
        return None

    logger.info(f"--- Running Step 4d: Statement Type ID (Agent: {statement_type_identifier_agent.name}) ---")
    print(f"\n--- Running Step 4d: Statement Type ID using model: {STATEMENT_TYPE_MODEL} ---")

    step4d_metadata_for_trace = {
        "workflow_step": "4d_statement_type_id",
        "agent_name": "Statement Type ID",
        "actual_agent": str(statement_type_identifier_agent.name),
        "primary_domain_input": primary_domain,
        "sub_domains_analyzed_count": str(len(sub_domain_data.identified_sub_domains)),
        "topics_aggregated_count": str(sum(len(t.identified_topics) for t in topic_data.sub_domain_topic_map)),
    }
    step4d_run_config = RunConfig(trace_metadata={k: str(v) for k, v in step4d_metadata_for_trace.items()})
    step4d_result: Optional[RunResult] = None
    statement_data: Optional[StatementTypeSchema] = None

    # Prepare context summary for the prompt
    context_summary_for_prompt = (
        f"Primary Domain: {primary_domain}\n"
        f"Identified Sub-Domains: {', '.join(sd.sub_domain for sd in sub_domain_data.identified_sub_domains)}\n"
        f"Previously identified topics (aggregated): {len(topic_data.sub_domain_topic_map)} sub-domains covered with topics."
        # Optionally add more topic detail here if needed
    )

    step4d_input_list: List[TResponseInputItem] = [
        {"role": "user", "content": (
            f"Analyze the following text to identify key STATEMENT types (e.g., Fact, Claim, Opinion, Question, Instruction). "
            f"Use the provided context:\n{context_summary_for_prompt}\n\n"
            f"Identify statement types relevant to this overall context. "
            f"Output ONLY using the required StatementTypeSchema, including the primary_domain and analyzed_sub_domains list in the output."
        )},
        {"role": "user", "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---"}
    ]

    try:
        step4d_result = await run_agent_with_retry(
            agent=statement_type_identifier_agent,
            input_data=step4d_input_list,
            config=step4d_run_config
        )

        if step4d_result:
            potential_output_step4d = getattr(step4d_result, 'final_output', None)
            if isinstance(potential_output_step4d, StatementTypeSchema):
                statement_data = potential_output_step4d
                logger.info("Successfully extracted StatementTypeSchema from step4d_result.final_output.")
            elif isinstance(potential_output_step4d, dict):
                try:
                    statement_data = StatementTypeSchema.model_validate(potential_output_step4d)
                    logger.info("Successfully validated StatementTypeSchema from step4d_result.final_output dict.")
                except ValidationError as e:
                    logger.warning(f"Step 4d dict output failed StatementTypeSchema validation: {e}")
            else:
                logger.warning(f"Step 4d final_output was not StatementTypeSchema or dict (type: {type(potential_output_step4d)}).")

            if statement_data and statement_data.identified_statements:
                # Ensure context fields match
                if statement_data.primary_domain != primary_domain:
                    logger.warning(f"Primary domain mismatch in Step 4d output ('{statement_data.primary_domain}'). Overwriting with Step 1's ('{primary_domain}').")
                    statement_data.primary_domain = primary_domain
                if set(statement_data.analyzed_sub_domains) != set(sd.sub_domain for sd in sub_domain_data.identified_sub_domains):
                    logger.warning(f"Analyzed sub-domains in Step 4d output {statement_data.analyzed_sub_domains} differs from Step 2 input { [sd.sub_domain for sd in sub_domain_data.identified_sub_domains]}. Using Step 4d's list.")

                # Log and print results
                statement_log_items = [item.statement_type for item in statement_data.identified_statements]
                logger.info(f"Step 4d Result: Identified Statement Types = [{', '.join(statement_log_items)}]")
                logger.info(f"Step 4d Result (Structured Statements):\n{statement_data.model_dump_json(indent=2)}")
                print("\n--- Statement Types Identified (Structured Output from Step 4d) ---")
                print(statement_data.model_dump_json(indent=2))

                # Save results
                logger.info("Saving statement type identifier output to file...")
                print("\nSaving statement type output file...")
                statement_type_output_content = {
                    "primary_domain": statement_data.primary_domain,
                    "analyzed_sub_domains": statement_data.analyzed_sub_domains,
                    "identified_statements": [item.model_dump() for item in statement_data.identified_statements],
                    "analysis_summary": statement_data.analysis_summary,
                    "analysis_details": {
                        "source_text_length": len(content),
                        "primary_domain_context": primary_domain,
                        "sub_domain_context_count": len(sub_domain_data.identified_sub_domains),
                        "topic_context_count": sum(len(t.identified_topics) for t in topic_data.sub_domain_topic_map),
                        "model_used": STATEMENT_TYPE_MODEL,
                        "agent_name": statement_type_identifier_agent.name,
                        "output_schema": StatementTypeSchema.__name__,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat()
                    },
                    "trace_information": {
                        "trace_id": overall_trace_id or "N/A",
                        "notes": f"Generated by {statement_type_identifier_agent.name} in Step 4d of workflow."
                    }
                }
                save_result_step4d = direct_save_json_output(
                    STATEMENT_TYPE_OUTPUT_DIR, STATEMENT_TYPE_OUTPUT_FILENAME, statement_type_output_content, overall_trace_id
                )
                print(f"  - {save_result_step4d}")
                logger.info(f"Result of saving statement type output: {save_result_step4d}")

            elif statement_data and not statement_data.identified_statements:
                logger.warning("Step 4d completed but identified_statements list is empty.")
                print("\nStep 4d completed, but no specific statement types were identified.")
                # Don't signal failure unless needed
                # statement_data = None
            else: # statement_data is None or validation failed
                logger.error("Step 4d FAILED: Could not extract valid StatementTypeSchema output.")
                print("\nError: Failed to identify statement types in Step 4d.")
                statement_data = None # Signal failure if needed

        else:
            logger.error("Step 4d FAILED: Runner.run did not return a result.")
            print("\nError: Failed to get a result from the statement type identification step.")
            statement_data = None

    except (ValidationError, TypeError) as e:
        logger.exception(f"Validation or Type error during Step 4d agent run. Error: {e}", extra={"trace_id": overall_trace_id or 'N/A'})
        print(f"\nError: A data validation or type issue occurred during Step 4d.")
        print(f"Error details: {e}")
        statement_data = None
    except Exception as e:
        logger.exception("An unexpected error occurred during Step 4d.", extra={"trace_id": overall_trace_id or 'N/A'})
        print(f"\nAn unexpected error occurred during Step 4d: {type(e).__name__}: {e}")
        statement_data = None

    return statement_data