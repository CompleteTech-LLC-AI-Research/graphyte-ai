"""Step 4b: Ontology type identification functionality."""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import ValidationError

from agents import RunConfig, RunResult, TResponseInputItem  # type: ignore[attr-defined]

from ..workflow_agents import ontology_type_identifier_agent
from ..config import (
    ONTOLOGY_TYPE_MODEL,
    ONTOLOGY_TYPE_OUTPUT_DIR,
    ONTOLOGY_TYPE_OUTPUT_FILENAME,
)
from ..schemas import OntologyTypeSchema, SubDomainSchema, TopicSchema
from ..utils import direct_save_json_output, run_agent_with_retry

logger = logging.getLogger(__name__)


async def identify_ontology_types(
    content: str,
    primary_domain: str,
    sub_domain_data: SubDomainSchema,
    topic_data: TopicSchema,
    trace_id: Optional[str] = None,
    group_id: Optional[str] = None,
) -> Optional[OntologyTypeSchema]:
    """Identify ontology types based on domain, sub-domains, and topics.

    Args:
        content: The text content to analyze
        primary_domain: The primary domain identified in step 1
        sub_domain_data: The SubDomainSchema from step 2
        topic_data: The TopicSchema from step 3
        trace_id: The trace ID for logging purposes
        group_id: The trace group ID for logging purposes

    Returns:
        An OntologyTypeSchema object if successful, None otherwise
    """
    if not primary_domain or not sub_domain_data or not topic_data:
        logger.info("Skipping Step 4b because prerequisites were not identified.")
        if not primary_domain:
            print("Skipping Step 4b as primary domain was not identified.")
        elif not sub_domain_data:
            print("Skipping Step 4b as sub-domain identification failed.")
        elif not topic_data:
            print("Skipping Step 4b as topic identification failed.")
        return None

    logger.info(
        f"--- Running Step 4b: Ontology Type ID (Agent: {ontology_type_identifier_agent.name}) ---"
    )
    print(
        f"\n--- Running Step 4b: Ontology Type ID using model: {ONTOLOGY_TYPE_MODEL} ---"
    )

    step4b_metadata_for_trace = {
        "workflow_step": "4b_ontology_type_id",
        "agent_name": "Ontology Type ID",
        "actual_agent": str(ontology_type_identifier_agent.name),
        "primary_domain_input": primary_domain,
        "sub_domains_analyzed_count": str(len(sub_domain_data.identified_sub_domains)),
        "topics_aggregated_count": str(
            sum(len(t.identified_topics) for t in topic_data.sub_domain_topic_map)
        ),
    }
    step4b_run_config = RunConfig(
        workflow_name="step4b_ontology_types",
        trace_id=trace_id,
        group_id=group_id,
        trace_metadata={k: str(v) for k, v in step4b_metadata_for_trace.items()},
    )
    step4b_result: Optional[RunResult] = None
    ontology_data: Optional[OntologyTypeSchema] = None

    # Prepare context summary for the prompt
    context_summary_for_prompt = (
        f"Primary Domain: {primary_domain}\n"
        f"Identified Sub-Domains: {', '.join(sd.sub_domain for sd in sub_domain_data.identified_sub_domains)}\n"
        f"Previously identified topics (aggregated): {len(topic_data.sub_domain_topic_map)} sub-domains covered with topics."
    )

    step4b_input_list: List[TResponseInputItem] = [
        {
            "role": "user",
            "content": (
                f"Analyze the following text to identify relevant ontology types or concepts, potentially referencing standard ontologies "
                f"(like Schema.org, FIBO, domain-specific ones) where applicable. "
                f"Use the provided context:\n{context_summary_for_prompt}\n\n"
                f"Identify ontology types/concepts relevant to this overall context. "
                f"Output ONLY using the required OntologyTypeSchema, including the primary_domain and analyzed_sub_domains list in the output."
            ),
        },
        {
            "role": "user",
            "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---",
        },
    ]

    try:
        step4b_result = await run_agent_with_retry(
            agent=ontology_type_identifier_agent,
            input_data=step4b_input_list,
            config=step4b_run_config,
        )

        if step4b_result:
            potential_output_step4b = getattr(step4b_result, "final_output", None)
            if isinstance(potential_output_step4b, OntologyTypeSchema):
                ontology_data = potential_output_step4b
                logger.info(
                    "Successfully extracted OntologyTypeSchema from step4b_result.final_output."
                )
            elif isinstance(potential_output_step4b, dict):
                try:
                    ontology_data = OntologyTypeSchema.model_validate(
                        potential_output_step4b
                    )
                    logger.info(
                        "Successfully validated OntologyTypeSchema from step4b_result.final_output dict."
                    )
                except ValidationError as e:
                    logger.warning(
                        f"Step 4b dict output failed OntologyTypeSchema validation: {e}"
                    )
            else:
                logger.warning(
                    f"Step 4b final_output was not OntologyTypeSchema or dict (type: {type(potential_output_step4b)})."
                )

            if ontology_data and ontology_data.identified_ontology_types:
                # Ensure context fields in output schema match expectations
                if ontology_data.primary_domain != primary_domain:
                    logger.warning(
                        f"Primary domain mismatch in Step 4b output ('{ontology_data.primary_domain}'). Overwriting with Step 1's ('{primary_domain}')."
                    )
                    ontology_data.primary_domain = primary_domain

                # Check sub-domains match input context (similar to entity types)
                if set(ontology_data.analyzed_sub_domains) != set(
                    sd.sub_domain for sd in sub_domain_data.identified_sub_domains
                ):
                    logger.warning(
                        f"Analyzed sub-domains in Step 4b output {ontology_data.analyzed_sub_domains} differs from Step 2 input { [sd.sub_domain for sd in sub_domain_data.identified_sub_domains]}. Using Step 4b's list."
                    )

                # Log and print results
                ontology_log_items = [
                    item.ontology_type
                    for item in ontology_data.identified_ontology_types
                ]
                logger.info(
                    f"Step 4b Result: Identified Ontology Types = [{', '.join(ontology_log_items)}]"
                )
                logger.info(
                    f"Step 4b Result (Structured Ontology Types):\n{ontology_data.model_dump_json(indent=2)}"
                )
                print(
                    "\n--- Ontology Types Identified (Structured Output from Step 4b) ---"
                )
                print(ontology_data.model_dump_json(indent=2))

                # Save results
                logger.info("Saving ontology type identifier output to file...")
                print("\nSaving ontology type output file...")
                ontology_type_output_content = {
                    "primary_domain": ontology_data.primary_domain,
                    "analyzed_sub_domains": ontology_data.analyzed_sub_domains,  # Use agent's output list
                    "identified_ontology_types": [
                        item.model_dump()
                        for item in ontology_data.identified_ontology_types
                    ],
                    "analysis_summary": ontology_data.analysis_summary,
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
                        "model_used": ONTOLOGY_TYPE_MODEL,
                        "agent_name": ontology_type_identifier_agent.name,
                        "output_schema": OntologyTypeSchema.__name__,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    "trace_information": {
                        "trace_id": trace_id or "N/A",
                        "notes": f"Generated by {ontology_type_identifier_agent.name} in Step 4b of workflow.",
                    },
                }
                save_result_step4b = direct_save_json_output(
                    ONTOLOGY_TYPE_OUTPUT_DIR,
                    ONTOLOGY_TYPE_OUTPUT_FILENAME,
                    ontology_type_output_content,
                    trace_id,
                )
                print(f"  - {save_result_step4b}")
                logger.info(
                    f"Result of saving ontology type output: {save_result_step4b}"
                )

            elif ontology_data and not ontology_data.identified_ontology_types:
                logger.warning(
                    "Step 4b completed but identified_ontology_types list is empty."
                )
                print(
                    "\nStep 4b completed, but no specific ontology types were identified."
                )
                ontology_data = None  # Signal failure for next step
            else:  # ontology_data is None or validation failed
                logger.error(
                    "Step 4b FAILED: Could not extract valid OntologyTypeSchema output."
                )
                print("\nError: Failed to identify ontology types in Step 4b.")
                ontology_data = None  # Signal failure for next step

        else:
            logger.error("Step 4b FAILED: Runner.run did not return a result.")
            print(
                "\nError: Failed to get a result from the ontology type identification step."
            )
            ontology_data = None

    except (ValidationError, TypeError) as e:
        logger.exception(
            f"Validation or Type error during Step 4b agent run. Error: {e}",
            extra={"trace_id": trace_id or "N/A"},
        )
        print("\nError: A data validation or type issue occurred during Step 4b.")
        print(f"Error details: {e}")
        ontology_data = None
    except Exception as e:
        logger.exception(
            "An unexpected error occurred during Step 4b.",
            extra={"trace_id": trace_id or "N/A"},
        )
        print(f"\nAn unexpected error occurred during Step 4b: {type(e).__name__}: {e}")
        ontology_data = None

    return ontology_data
