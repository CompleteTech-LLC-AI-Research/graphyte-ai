"""Step 5: Statement instance extraction functionality."""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import ValidationError
from agentic_team import RunConfig, RunResult, TResponseInputItem

from ..agents import statement_instance_extractor_agent
from ..config import STATEMENT_INSTANCE_MODEL, STATEMENT_INSTANCE_OUTPUT_DIR, STATEMENT_INSTANCE_OUTPUT_FILENAME
from ..schemas import (
    StatementInstanceSchema, SubDomainSchema, TopicSchema, StatementTypeSchema
)
from ..utils import direct_save_json_output, run_agent_with_retry

logger = logging.getLogger(__name__)


async def identify_statement_instances(
    content: str,
    primary_domain: str,
    sub_domain_data: SubDomainSchema,
    topic_data: TopicSchema,
    statement_type_data: StatementTypeSchema,
    overall_trace_id: Optional[str] = None
) -> Optional[StatementInstanceSchema]:
    """Extract specific statements with their classified types."""
    if not (primary_domain and sub_domain_data and topic_data and statement_type_data):
        logger.info("Skipping Step 5 (Statement Instances) because prerequisites were not met.")
        return None

    logger.info(
        f"--- Running Step 5: Statement Instance Extraction (Agent: {statement_instance_extractor_agent.name}) ---"
    )
    print(f"\n--- Running Step 5: Statement Instance Extraction using model: {STATEMENT_INSTANCE_MODEL} ---")

    metadata = {
        "workflow_step": "5_statement_instance_extraction",
        "agent_name": "Statement Instance Extraction",
        "actual_agent": str(statement_instance_extractor_agent.name),
        "primary_domain_input": primary_domain,
        "sub_domains_analyzed_count": str(len(sub_domain_data.identified_sub_domains)),
        "topics_aggregated_count": str(sum(len(t.identified_topics) for t in topic_data.sub_domain_topic_map)),
        "statement_types_count": str(len(statement_type_data.identified_statements)),
    }
    run_config = RunConfig(trace_metadata={k: str(v) for k, v in metadata.items()})

    context_summary = (
        f"Primary Domain: {primary_domain}\n"
        f"Identified Sub-Domains: {', '.join(sd.sub_domain for sd in sub_domain_data.identified_sub_domains)}\n"
        f"Statement Types Identified: {', '.join(st.statement_type for st in statement_type_data.identified_statements)}"
    )

    input_list: List[TResponseInputItem] = [
        {
            "role": "user",
            "content": (
                "Extract distinct statements from the text and classify each with its statement type. "
                f"Use the provided context:\n{context_summary}\n\n"
                "Output ONLY using the required StatementInstanceSchema."
            ),
        },
        {"role": "user", "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---"},
    ]

    instance_data: Optional[StatementInstanceSchema] = None
    try:
        result: Optional[RunResult] = await run_agent_with_retry(
            agent=statement_instance_extractor_agent,
            input_data=input_list,
            config=run_config,
        )

        if result:
            potential_output = getattr(result, "final_output", None)
            if isinstance(potential_output, StatementInstanceSchema):
                instance_data = potential_output
            elif isinstance(potential_output, dict):
                try:
                    instance_data = StatementInstanceSchema.model_validate(potential_output)
                except ValidationError as e:
                    logger.warning(f"Dict output failed StatementInstanceSchema validation: {e}")
            else:
                logger.warning(
                    f"Step 5 output was not StatementInstanceSchema or dict (type: {type(potential_output)})"
                )

            if instance_data and instance_data.statement_instances:
                if instance_data.primary_domain != primary_domain:
                    instance_data.primary_domain = primary_domain
                print("\n--- Statement Instances Extracted ---")
                print(instance_data.model_dump_json(indent=2))

                output_content = {
                    "primary_domain": instance_data.primary_domain,
                    "analyzed_sub_domains": instance_data.analyzed_sub_domains,
                    "statement_instances": [s.model_dump() for s in instance_data.statement_instances],
                    "analysis_summary": instance_data.analysis_summary,
                    "analysis_details": {
                        "source_text_length": len(content),
                        "primary_domain_context": primary_domain,
                        "sub_domain_context_count": len(sub_domain_data.identified_sub_domains),
                        "topic_context_count": sum(len(t.identified_topics) for t in topic_data.sub_domain_topic_map),
                        "statement_type_context_count": len(statement_type_data.identified_statements),
                        "model_used": STATEMENT_INSTANCE_MODEL,
                        "agent_name": statement_instance_extractor_agent.name,
                        "output_schema": StatementInstanceSchema.__name__,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    "trace_information": {
                        "trace_id": overall_trace_id or "N/A",
                        "notes": f"Generated by {statement_instance_extractor_agent.name} in Step 5 of workflow.",
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
            elif instance_data and not instance_data.statement_instances:
                logger.warning("Step 5 completed but no statement instances identified.")
                print("\nStep 5 completed, but no statement instances were identified.")
                instance_data = None
            else:
                logger.error("Step 5 FAILED: Could not extract valid StatementInstanceSchema output.")
                print("\nError: Failed to extract statement instances in Step 5.")
                instance_data = None
        else:
            logger.error("Step 5 FAILED: Runner.run did not return a result.")
            print("\nError: Failed to get a result from the statement instance extraction step.")
    except (ValidationError, TypeError) as e:
        logger.exception(
            f"Validation or Type error during Step 5 agent run. Error: {e}", extra={"trace_id": overall_trace_id or 'N/A'}
        )
        print("\nError: A data validation or type issue occurred during Step 5.")
        print(f"Error details: {e}")
    except Exception as e:
        logger.exception("An unexpected error occurred during Step 5.", extra={"trace_id": overall_trace_id or 'N/A'})
        print(f"\nAn unexpected error occurred during Step 5: {type(e).__name__}: {e}")

    return instance_data

