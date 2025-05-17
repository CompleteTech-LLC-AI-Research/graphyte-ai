"""Aggregate outputs of Step 5 instance extraction agents."""

import logging
from datetime import datetime, timezone
from typing import Optional

from ..schemas import (
    ExtractedInstancesSchema,
    SubDomainSchema,
    EntityInstanceSchema,
    OntologyInstanceSchema,
    EventInstanceSchema,
    StatementInstanceSchema,
    EvidenceInstanceSchema,
    MeasurementInstanceSchema,
    ModalityInstanceSchema,
)
from ..config import (
    AGGREGATED_INSTANCE_OUTPUT_DIR,
    AGGREGATED_INSTANCE_OUTPUT_FILENAME,
)
from ..utils import direct_save_json_output

logger = logging.getLogger(__name__)


def aggregate_extracted_instances(
    primary_domain: str,
    sub_domain_data: SubDomainSchema,
    entity_instances: Optional[EntityInstanceSchema] = None,
    ontology_instances: Optional[OntologyInstanceSchema] = None,
    event_instances: Optional[EventInstanceSchema] = None,
    statement_instances: Optional[StatementInstanceSchema] = None,
    evidence_instances: Optional[EvidenceInstanceSchema] = None,
    measurement_instances: Optional[MeasurementInstanceSchema] = None,
    modality_instances: Optional[ModalityInstanceSchema] = None,
    overall_trace_id: Optional[str] = None,
) -> Optional[ExtractedInstancesSchema]:
    """Combine individual instance outputs from Steps 5a–5g."""

    if not primary_domain or not sub_domain_data:
        logger.warning(
            "aggregate_extracted_instances skipped due to missing required context."
        )
        return None

    aggregated = ExtractedInstancesSchema(
        primary_domain=primary_domain,
        analyzed_sub_domains=[sd.sub_domain for sd in sub_domain_data.identified_sub_domains],
        entity_instances=entity_instances.identified_instances if entity_instances else [],
        ontology_instances=ontology_instances.identified_instances if ontology_instances else [],
        event_instances=event_instances.identified_instances if event_instances else [],
        statement_instances=statement_instances.identified_instances if statement_instances else [],
        evidence_instances=evidence_instances.identified_instances if evidence_instances else [],
        measurement_instances=measurement_instances.identified_instances if measurement_instances else [],
        modality_instances=modality_instances.identified_instances if modality_instances else [],
        analysis_summary=(
            "Aggregated instance results from Steps 5a-5g."
        ),
    )

    logger.info(
        "Aggregated extracted instances:\n%s",
        aggregated.model_dump_json(indent=2),
    )
    print("\n--- Aggregated Extracted Instances ---")
    print(aggregated.model_dump_json(indent=2))

    output_content = {
        "primary_domain": aggregated.primary_domain,
        "analyzed_sub_domains": aggregated.analyzed_sub_domains,
        "entity_instances": [i.model_dump() for i in aggregated.entity_instances],
        "ontology_instances": [i.model_dump() for i in aggregated.ontology_instances],
        "event_instances": [i.model_dump() for i in aggregated.event_instances],
        "statement_instances": [i.model_dump() for i in aggregated.statement_instances],
        "evidence_instances": [i.model_dump() for i in aggregated.evidence_instances],
        "measurement_instances": [i.model_dump() for i in aggregated.measurement_instances],
        "modality_instances": [i.model_dump() for i in aggregated.modality_instances],
        "analysis_summary": aggregated.analysis_summary,
        "analysis_details": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "trace_information": {
            "trace_id": overall_trace_id or "N/A",
            "notes": "Aggregated from instance extraction steps",
        },
    }

    save_result = direct_save_json_output(
        AGGREGATED_INSTANCE_OUTPUT_DIR,
        AGGREGATED_INSTANCE_OUTPUT_FILENAME,
        output_content,
        overall_trace_id,
    )
    print(f"  - {save_result}")
    logger.info(f"Result of saving aggregated instance output: {save_result}")

    return aggregated
