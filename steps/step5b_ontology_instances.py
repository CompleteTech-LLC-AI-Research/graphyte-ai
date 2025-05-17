"""Step 5b: Ontology instance extraction functionality."""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import ValidationError

from agentic_team import RunConfig, RunResult, TResponseInputItem

from ..agents import ontology_instance_extractor_agent
from ..config import ONTOLOGY_INSTANCE_MODEL, ONTOLOGY_INSTANCE_OUTPUT_DIR, ONTOLOGY_INSTANCE_OUTPUT_FILENAME
from ..schemas import OntologyInstanceSchema, SubDomainSchema, TopicSchema, OntologyTypeSchema
from ..utils import direct_save_json_output, run_agent_with_retry

logger = logging.getLogger(__name__)

async def identify_ontology_instances(
    content: str,
    primary_domain: str,
    sub_domain_data: SubDomainSchema,
    topic_data: TopicSchema,
    ontology_data: OntologyTypeSchema,
    overall_trace_id: Optional[str] = None,
) -> Optional[OntologyInstanceSchema]:
    """Extract ontology concept mentions from the text based on context."""
    if not primary_domain or not sub_domain_data or not topic_data or not ontology_data:
        logger.info("Skipping Step 5b because prerequisites were not identified.")
        if not primary_domain:
            print("Skipping Step 5b as primary domain was not identified.")
        elif not sub_domain_data:
            print("Skipping Step 5b as sub-domain identification failed.")
        elif not topic_data:
            print("Skipping Step 5b as topic identification failed.")
        elif not ontology_data:
            print("Skipping Step 5b as ontology type identification failed.")
        return None

    logger.info(
        f"--- Running Step 5b: Ontology Instance Extraction (Agent: {ontology_instance_extractor_agent.name}) ---"
    )
    print(f"\n--- Running Step 5b: Ontology Instance Extraction using model: {ONTOLOGY_INSTANCE_MODEL} ---")

    step5b_metadata_for_trace = {
        "workflow_step": "5b_ontology_instance_extraction",
        "agent_name": "Ontology Instance Extractor",
        "actual_agent": str(ontology_instance_extractor_agent.name),
        "primary_domain_input": primary_domain,
        "sub_domains_analyzed_count": str(len(sub_domain_data.identified_sub_domains)),
        "topic_context_count": str(sum(len(t.identified_topics) for t in topic_data.sub_domain_topic_map)),
        "ontology_type_count": str(len(ontology_data.identified_ontology_types)),
    }
    step5b_run_config = RunConfig(trace_metadata={k: str(v) for k, v in step5b_metadata_for_trace.items()})
    step5b_result: Optional[RunResult] = None
    instance_data: Optional[OntologyInstanceSchema] = None

    context_summary_for_prompt = (
        f"Primary Domain: {primary_domain}\n"
        f"Identified Sub-Domains: {', '.join(sd.sub_domain for sd in sub_domain_data.identified_sub_domains)}\n"
        f"Ontology Types Considered: {', '.join(o.ontology_type for o in ontology_data.identified_ontology_types)}"
    )

    step5b_input_list: List[TResponseInputItem] = [
        {
            "role": "user",
            "content": (
                f"Extract specific mentions of ontology concepts from the text. Use the provided context:\n{context_summary_for_prompt}\n\n"
                f"Provide the ontology type, exact text span and character offsets. Output ONLY using the required OntologyInstanceSchema."
            ),
        },
        {"role": "user", "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---"},
    ]

    try:
        step5b_result = await run_agent_with_retry(
            agent=ontology_instance_extractor_agent,
            input_data=step5b_input_list,
            config=step5b_run_config,
        )

        if step5b_result:
            potential_output = getattr(step5b_result, "final_output", None)
            if isinstance(potential_output, OntologyInstanceSchema):
                instance_data = potential_output
            elif isinstance(potential_output, dict):
                try:
                    instance_data = OntologyInstanceSchema.model_validate(potential_output)
                except ValidationError as e:
                    logger.warning(f"Step 5b dict output failed OntologyInstanceSchema validation: {e}")
            else:
                logger.warning(
                    f"Step 5b final_output was not OntologyInstanceSchema or dict (type: {type(potential_output)})."
                )

            if instance_data and instance_data.identified_instances:
                if instance_data.primary_domain != primary_domain:
                    instance_data.primary_domain = primary_domain
                if not set(instance_data.analyzed_sub_domains):
                    instance_data.analyzed_sub_domains = [sd.sub_domain for sd in sub_domain_data.identified_sub_domains]
                logger.info(
                    f"Step 5b Result (Structured Instances):\n{instance_data.model_dump_json(indent=2)}"
                )
                print("\n--- Ontology Instances Extracted (Structured Output) ---")
                print(instance_data.model_dump_json(indent=2))

                output_content = {
                    "primary_domain": instance_data.primary_domain,
                    "analyzed_sub_domains": instance_data.analyzed_sub_domains,
                    "analyzed_ontology_types": instance_data.analyzed_ontology_types,
                    "identified_instances": [item.model_dump() for item in instance_data.identified_instances],
                    "analysis_summary": instance_data.analysis_summary,
                    "analysis_details": {
                        "source_text_length": len(content),
                        "model_used": ONTOLOGY_INSTANCE_MODEL,
                        "agent_name": ontology_instance_extractor_agent.name,
                        "output_schema": OntologyInstanceSchema.__name__,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    "trace_information": {
                        "trace_id": overall_trace_id or "N/A",
                        "notes": f"Generated by {ontology_instance_extractor_agent.name} in Step 5b of workflow.",
                    },
                }
                save_result = direct_save_json_output(
                    ONTOLOGY_INSTANCE_OUTPUT_DIR,
                    ONTOLOGY_INSTANCE_OUTPUT_FILENAME,
                    output_content,
                    overall_trace_id,
                )
                print(f"  - {save_result}")
                logger.info(f"Result of saving ontology instance output: {save_result}")
            elif instance_data and not instance_data.identified_instances:
                logger.warning("Step 5b completed but identified_instances list is empty.")
                print("\nStep 5b completed, but no ontology instances were identified.")
            else:
                logger.error("Step 5b FAILED: Could not extract valid OntologyInstanceSchema output.")
                print("\nError: Failed to extract ontology instances in Step 5b.")
                instance_data = None
        else:
            logger.error("Step 5b FAILED: Runner.run did not return a result.")
            print("\nError: Failed to get a result from the ontology instance extraction step.")
            instance_data = None

    except (ValidationError, TypeError) as e:
        logger.exception(
            f"Validation or Type error during Step 5b agent run. Error: {e}", extra={"trace_id": overall_trace_id or 'N/A'}
        )
        print("\nError: A data validation or type issue occurred during Step 5b.")
        print(f"Error details: {e}")
        instance_data = None
    except Exception as e:
        logger.exception("An unexpected error occurred during Step 5b.", extra={"trace_id": overall_trace_id or 'N/A'})
        print(f"\nAn unexpected error occurred during Step 5b: {type(e).__name__}: {e}")
        instance_data = None

    return instance_data
