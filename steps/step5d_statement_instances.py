"""Step 5d: Statement instance extraction functionality."""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import ValidationError

from agentic_team import RunConfig, RunResult, TResponseInputItem

from ..agents import statement_instance_extractor_agent
from ..config import STATEMENT_INSTANCE_MODEL, STATEMENT_INSTANCE_OUTPUT_DIR, STATEMENT_INSTANCE_OUTPUT_FILENAME
from ..schemas import StatementInstanceSchema, SubDomainSchema, TopicSchema, StatementTypeSchema
from ..utils import direct_save_json_output, run_agent_with_retry

logger = logging.getLogger(__name__)

async def identify_statement_instances(
    content: str,
    primary_domain: str,
    sub_domain_data: SubDomainSchema,
    topic_data: TopicSchema,
    statement_data: StatementTypeSchema,
    overall_trace_id: Optional[str] = None,
) -> Optional[StatementInstanceSchema]:
    """Extract statement mentions from the text based on context."""
    if not primary_domain or not sub_domain_data or not topic_data or not statement_data:
        logger.info("Skipping Step 5d because prerequisites were not identified.")
        if not primary_domain:
            print("Skipping Step 5d as primary domain was not identified.")
        elif not sub_domain_data:
            print("Skipping Step 5d as sub-domain identification failed.")
        elif not topic_data:
            print("Skipping Step 5d as topic identification failed.")
        elif not statement_data:
            print("Skipping Step 5d as statement type identification failed.")
        return None

    logger.info(
        f"--- Running Step 5d: Statement Instance Extraction (Agent: {statement_instance_extractor_agent.name}) ---"
    )
    print(f"\n--- Running Step 5d: Statement Instance Extraction using model: {STATEMENT_INSTANCE_MODEL} ---")

    step5d_metadata_for_trace = {
        "workflow_step": "5d_statement_instance_extraction",
        "agent_name": "Statement Instance Extractor",
        "actual_agent": str(statement_instance_extractor_agent.name),
        "primary_domain_input": primary_domain,
        "sub_domains_analyzed_count": str(len(sub_domain_data.identified_sub_domains)),
        "topic_context_count": str(sum(len(t.identified_topics) for t in topic_data.sub_domain_topic_map)),
        "statement_type_count": str(len(statement_data.identified_statements)),
    }
    step5d_run_config = RunConfig(trace_metadata={k: str(v) for k, v in step5d_metadata_for_trace.items()})
    step5d_result: Optional[RunResult] = None
    instance_data: Optional[StatementInstanceSchema] = None

    context_summary_for_prompt = (
        f"Primary Domain: {primary_domain}\n"
        f"Identified Sub-Domains: {', '.join(sd.sub_domain for sd in sub_domain_data.identified_sub_domains)}\n"
        f"Statement Types Considered: {', '.join(s.statement_type for s in statement_data.identified_statements)}"
    )

    step5d_input_list: List[TResponseInputItem] = [
        {
            "role": "user",
            "content": (
                f"Extract specific statements from the text. Use the provided context:\n{context_summary_for_prompt}\n\n"
                f"Provide the statement type, exact text span and character offsets. Output ONLY using the required StatementInstanceSchema."
            ),
        },
        {"role": "user", "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---"},
    ]

    try:
        step5d_result = await run_agent_with_retry(
            agent=statement_instance_extractor_agent,
            input_data=step5d_input_list,
            config=step5d_run_config,
        )

        if step5d_result:
            potential_output = getattr(step5d_result, "final_output", None)
            if isinstance(potential_output, StatementInstanceSchema):
                instance_data = potential_output
            elif isinstance(potential_output, dict):
                try:
                    instance_data = StatementInstanceSchema.model_validate(potential_output)
                except ValidationError as e:
                    logger.warning(f"Step 5d dict output failed StatementInstanceSchema validation: {e}")
            else:
                logger.warning(
                    f"Step 5d final_output was not StatementInstanceSchema or dict (type: {type(potential_output)})."
                )

            if instance_data and instance_data.identified_instances:
                if instance_data.primary_domain != primary_domain:
                    instance_data.primary_domain = primary_domain
                if not set(instance_data.analyzed_sub_domains):
                    instance_data.analyzed_sub_domains = [sd.sub_domain for sd in sub_domain_data.identified_sub_domains]
                logger.info(
                    f"Step 5d Result (Structured Instances):\n{instance_data.model_dump_json(indent=2)}"
                )
                print("\n--- Statement Instances Extracted (Structured Output) ---")
                print(instance_data.model_dump_json(indent=2))

                output_content = {
                    "primary_domain": instance_data.primary_domain,
                    "analyzed_sub_domains": instance_data.analyzed_sub_domains,
                    "analyzed_statement_types": instance_data.analyzed_statement_types,
                    "identified_instances": [item.model_dump() for item in instance_data.identified_instances],
                    "analysis_summary": instance_data.analysis_summary,
                    "analysis_details": {
                        "source_text_length": len(content),
                        "model_used": STATEMENT_INSTANCE_MODEL,
                        "agent_name": statement_instance_extractor_agent.name,
                        "output_schema": StatementInstanceSchema.__name__,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    "trace_information": {
                        "trace_id": overall_trace_id or "N/A",
                        "notes": f"Generated by {statement_instance_extractor_agent.name} in Step 5d of workflow.",
                    },
                }
                save_result = direct_save_json_output(
                    STATEMENT_INSTANCE_OUTPUT_DIR,
                    STATEMENT_INSTANCE_OUTPUT_FILENAME,
                    output_content,
                    overall_trace_id,
                )
                print(f"  - {save_result}")
                logger.info(f"Result of saving statement instance output: {save_result}")
            elif instance_data and not instance_data.identified_instances:
                logger.warning("Step 5d completed but identified_instances list is empty.")
                print("\nStep 5d completed, but no statement instances were identified.")
            else:
                logger.error("Step 5d FAILED: Could not extract valid StatementInstanceSchema output.")
                print("\nError: Failed to extract statement instances in Step 5d.")
                instance_data = None
        else:
            logger.error("Step 5d FAILED: Runner.run did not return a result.")
            print("\nError: Failed to get a result from the statement instance extraction step.")
            instance_data = None

    except (ValidationError, TypeError) as e:
        logger.exception(
            f"Validation or Type error during Step 5d agent run. Error: {e}", extra={"trace_id": overall_trace_id or 'N/A'}
        )
        print("\nError: A data validation or type issue occurred during Step 5d.")
        print(f"Error details: {e}")
        instance_data = None
    except Exception as e:
        logger.exception("An unexpected error occurred during Step 5d.", extra={"trace_id": overall_trace_id or 'N/A'})
        print(f"\nAn unexpected error occurred during Step 5d: {type(e).__name__}: {e}")
        instance_data = None

    return instance_data
