"""Step 6b: Relationship instance extraction functionality."""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import ValidationError

from agentic_team import RunConfig, RunResult, TResponseInputItem

from ..agents import relationship_extractor_agent
from ..config import (
    RELATIONSHIP_INSTANCE_MODEL,
    RELATIONSHIP_INSTANCE_OUTPUT_DIR,
    RELATIONSHIP_INSTANCE_OUTPUT_FILENAME,
)
from ..schemas import (
    RelationshipInstanceSchema,
    SubDomainSchema,
    RelationshipSchema,
)
from ..utils import direct_save_json_output, run_agent_with_retry

logger = logging.getLogger(__name__)


async def identify_relationship_instances(
    content: str,
    primary_domain: str,
    sub_domain_data: SubDomainSchema,
    relationship_type_data: RelationshipSchema,
    overall_trace_id: Optional[str] = None,
) -> Optional[RelationshipInstanceSchema]:
    """Extract relationship instances using prior type results and instances."""

    if not (primary_domain and sub_domain_data and relationship_type_data):
        logger.info("Skipping Step 6b because prerequisites were not identified.")
        return None

    logger.info(
        f"--- Running Step 6b: Relationship Instance Extraction (Agent: {relationship_extractor_agent.name}) ---"
    )
    print(
        f"\n--- Running Step 6b: Relationship Instance Extraction using model: {RELATIONSHIP_INSTANCE_MODEL} ---"
    )

    meta = {
        "workflow_step": "6b_relationship_instance_extraction",
        "agent_name": "Relationship Instance Extractor",
    }
    run_config = RunConfig(trace_metadata=meta)

    rel_types = [
        rel.relationship_type
        for m in relationship_type_data.entity_relationships_map
        for rel in m.identified_relationships
    ]
    context_summary = (
        f"Primary Domain: {primary_domain}\n"
        f"Identified Sub-Domains: {', '.join(sd.sub_domain for sd in sub_domain_data.identified_sub_domains)}\n"
        f"Known Relationship Types: {', '.join(sorted(set(rel_types)))}"
    )

    inputs: List[TResponseInputItem] = [
        {
            "role": "user",
            "content": (
                "Identify explicit relationship instances in the text. "
                f"Use the provided context:\n{context_summary}\n\n"
                "For each instance provide subject, relationship type, object, relevance score and optional snippet. "
                "Output ONLY using the required RelationshipInstanceSchema."
            ),
        },
        {"role": "user", "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---"},
    ]

    try:
        result: Optional[RunResult] = await run_agent_with_retry(
            agent=relationship_extractor_agent,
            input_data=inputs,
            config=run_config,
        )

        final: Optional[RelationshipInstanceSchema] = None
        if result:
            data = getattr(result, "final_output", None)
            if isinstance(data, RelationshipInstanceSchema):
                final = data
            elif isinstance(data, dict):
                try:
                    final = RelationshipInstanceSchema.model_validate(data)
                except ValidationError as e:
                    logger.warning("Validation error for relationship instances: %s", e)
            else:
                logger.warning("Unexpected output type for relationship instances: %s", type(data))

        if final and final.identified_instances:
            if not final.analyzed_sub_domains:
                final.analyzed_sub_domains = [sd.sub_domain for sd in sub_domain_data.identified_sub_domains]
            if final.primary_domain != primary_domain:
                final.primary_domain = primary_domain
            logger.info("Step 6b result:\n%s", final.model_dump_json(indent=2))
            print("\n--- Relationship Instances Extracted (Structured Output) ---")
            print(final.model_dump_json(indent=2))

            output_content = {
                "primary_domain": final.primary_domain,
                "analyzed_sub_domains": final.analyzed_sub_domains,
                "identified_instances": [i.model_dump() for i in final.identified_instances],
                "analysis_summary": final.analysis_summary,
                "analysis_details": {
                    "source_text_length": len(content),
                    "model_used": RELATIONSHIP_INSTANCE_MODEL,
                    "agent_name": relationship_extractor_agent.name,
                    "output_schema": RelationshipInstanceSchema.__name__,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
                "trace_information": {"trace_id": overall_trace_id or "N/A"},
            }
            direct_save_json_output(
                RELATIONSHIP_INSTANCE_OUTPUT_DIR,
                RELATIONSHIP_INSTANCE_OUTPUT_FILENAME,
                output_content,
                overall_trace_id,
            )
        else:
            if final:
                logger.warning("Step 6b completed but no instances identified.")
                print("\nStep 6b completed, but no relationship instances were identified.")
            else:
                logger.error("Step 6b FAILED: Could not get valid output.")
                print("\nError: Failed to extract relationship instances in Step 6b.")
        return final

    except (ValidationError, TypeError) as e:
        logger.exception("Validation error during Step 6b: %s", e)
        print("\nError: A data validation or type issue occurred during Step 6b.")
        return None
    except Exception as e:
        logger.exception("Unexpected error during Step 6b", exc_info=e)
        print(f"\nAn unexpected error occurred during Step 6b: {type(e).__name__}: {e}")
        return None
