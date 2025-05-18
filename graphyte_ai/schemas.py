from typing import List, Optional
from pydantic import BaseModel, Field

# --- Schemas for Existing Agents (1-3) ---


# Schema for primary domain output (Agent 1)
class DomainSchema(BaseModel):
    """Schema defining the expected output: the primary domain."""

    domain: str = Field(
        description=(
            "The single, most relevant high-level domain identified in the text "
            "(e.g., Technology, Finance, Healthcare, Arts, Science, Entertainment, Sports, Politics)."
        )
    )


# Simple schema used when only a confidence score is needed
class ConfidenceScoreSchema(BaseModel):
    """Represents just a confidence score for a prior analysis."""

    confidence_score: float = Field(
        description="Confidence level (0.0 to 1.0) expressing certainty in the related analysis."
    )


# Simple schema used when only a relevance score is needed
class RelevanceScoreSchema(BaseModel):
    """Represents just a relevance score for a particular item."""

    relevance_score: float = Field(
        description=(
            "Relevance level (0.0 to 1.0) expressing how strongly the item relates "
            "to the provided context."
        )
    )


# Simple schema used when only a clarity score is needed
class ClarityScoreSchema(BaseModel):
    """Represents how clear a provided item is."""

    clarity_score: float = Field(
        description="Clarity level (0.0 to 1.0) indicating how understandable the item is."
    )
    item_id: Optional[str] = Field(
        None, description="Optional identifier for the evaluated item."
    )


# Nested schema for a sub-domain
class SubDomainDetail(BaseModel):
    """Represents a single identified sub-domain."""

    sub_domain: str = Field(
        description="The specific sub-domain identified within the text."
    )


# Schema for sub-domain analysis output (Agent 2)
class SubDomainSchema(BaseModel):
    """Schema defining the expected output for sub-domain analysis."""

    primary_domain: str = Field(
        description="The primary domain that was provided as input."
    )
    identified_sub_domains: List[SubDomainDetail] = Field(
        description="A list of specific sub-domains identified within the text related to the primary domain. Should not be empty if analysis was possible."
    )
    analysis_summary: Optional[str] = Field(
        None, description="A brief summary or commentary on the sub-domain analysis."
    )


# Nested schema for a topic
class TopicDetail(BaseModel):
    """Represents a single identified topic."""

    topic: str = Field(description="The specific topic identified within the text.")


# Schema for the output of the *single* topic identification agent call (Agent 3)
class SingleSubDomainTopicSchema(BaseModel):
    """Represents topics identified for a single sub-domain."""

    sub_domain: str = Field(description="The sub-domain being analyzed.")
    identified_topics: List[TopicDetail] = Field(
        description="A list of specific topics identified within the text related to this sub-domain."
    )


# Schema for the final, aggregated topic output (used for saving)
class TopicSchema(BaseModel):
    """Schema defining the final aggregated output for topic identification analysis."""

    primary_domain: str = Field(description="The overall primary domain provided.")
    sub_domain_topic_map: List[SingleSubDomainTopicSchema] = Field(
        description="A list mapping each analyzed sub-domain to its identified topics."
    )
    analysis_summary: Optional[str] = Field(
        None,
        description="Optional summary of the topic identification process across all sub-domains (can be generated or left null).",
    )


# --- Schemas for Step 4 Agents (4a, 4b, 4c, 4d, 4e, 4f, 4g) ---


# Nested schema for an entity type (Agent 4a)
class EntityTypeDetail(BaseModel):
    """Represents an entity type."""

    entity_type: str = Field(
        description=(
            "The classified type of the entity (e.g., PERSON, ORGANIZATION, LOCATION, "
            "DATE, MONEY, PRODUCT, TECHNOLOGY, SCIENTIFIC_CONCEPT, ECONOMIC_INDICATOR)."  # Removed EVENT as it's now separate
        )
    )


# Schema for entity type analysis output (Agent 4a)
class EntityTypeSchema(BaseModel):
    """Schema defining the expected output for entity type analysis (Step 4a)."""

    primary_domain: str = Field(
        description="The primary domain context provided for the analysis."
    )
    analyzed_sub_domains: List[str] = Field(
        description="The list of sub-domains used as context during entity identification."
    )
    identified_entities: List[EntityTypeDetail] = Field(
        description="A list of specific entity types identified within the text, relevant to the provided context."
    )
    analysis_summary: Optional[str] = Field(
        None,
        description="A brief summary or commentary on the entity identification analysis.",
    )


# Nested schema for an ontology type/concept (Agent 4b)
class OntologyTypeDetail(BaseModel):
    """Represents an ontology type or concept."""

    ontology_type: str = Field(
        description=(
            "The identified ontology type or concept (e.g., Schema.org:Person, FIBO:FinancialInstrument, GO:biological_process)."
        )
    )


# Schema for ontology type analysis output (Agent 4b)
class OntologyTypeSchema(BaseModel):
    """Schema defining the expected output for ontology type analysis (Step 4b)."""

    primary_domain: str = Field(
        description="The primary domain context provided for the analysis."
    )
    analyzed_sub_domains: List[str] = Field(
        description="The list of sub-domains used as context during ontology identification."
    )
    identified_ontology_types: List[OntologyTypeDetail] = Field(
        description="A list of specific ontology types or concepts identified within the text, relevant to the provided context."
    )
    analysis_summary: Optional[str] = Field(
        None,
        description="A brief summary or commentary on the ontology identification analysis.",
    )


# Nested schema for an event type (Agent 4c)
class EventDetail(BaseModel):
    """Represents an identified event type."""

    event_type: str = Field(
        description=(
            "The classified type of the event identified (e.g., Meeting, Acquisition, Conference, Product Launch, Election, Natural Disaster)."
        )
    )
    # Optional: Add description if needed, like 'Brief description of the event instance'
    # description: Optional[str] = Field(None, description="A brief description or name of the specific event instance.")


# Schema for event type analysis output (Agent 4c)
class EventTypeSchema(BaseModel):
    """Schema defining the expected output for event type analysis (Step 4c)."""

    primary_domain: str = Field(
        description="The primary domain context provided for the analysis."
    )
    analyzed_sub_domains: List[str] = Field(
        description="The list of sub-domains used as context during event identification."
    )
    identified_events: List[EventDetail] = Field(
        description="A list of specific event types identified within the text, relevant to the provided context."
    )
    analysis_summary: Optional[str] = Field(
        None,
        description="A brief summary or commentary on the event identification analysis.",
    )


# Nested schema for a statement type (Agent 4d)
class StatementDetail(BaseModel):
    """Represents an identified statement type."""

    statement_type: str = Field(
        description=(
            "The classified type of the statement identified (e.g., Fact, Claim, Opinion, Question, Instruction, Hypothesis, Prediction)."
        )
    )
    # Optional: Add a snippet of the text classified
    # supporting_text: Optional[str] = Field(None, description="The text snippet classified as this statement type.")


# Schema for statement type analysis output (Agent 4d)
class StatementTypeSchema(BaseModel):
    """Schema defining the expected output for statement type analysis (Step 4d)."""

    primary_domain: str = Field(
        description="The primary domain context provided for the analysis."
    )
    analyzed_sub_domains: List[str] = Field(
        description="The list of sub-domains used as context during statement identification."
    )
    identified_statements: List[StatementDetail] = Field(
        description="A list of specific statement types identified within the text, relevant to the provided context."
    )
    analysis_summary: Optional[str] = Field(
        None,
        description="A brief summary or commentary on the statement identification analysis.",
    )


# Nested schema for an evidence type (Agent 4e)
class EvidenceDetail(BaseModel):
    """Represents an identified evidence type."""

    evidence_type: str = Field(
        description=(
            "The classified type of evidence identified (e.g., Testimony, Document, Statistic, Anecdote, Expert Opinion, Observation, Example)."
        )
    )


# Schema for evidence type analysis output (Agent 4e)
class EvidenceTypeSchema(BaseModel):
    """Schema defining the expected output for evidence type analysis (Step 4e)."""

    primary_domain: str = Field(
        description="The primary domain context provided for the analysis."
    )
    analyzed_sub_domains: List[str] = Field(
        description="The list of sub-domains used as context during evidence identification."
    )
    identified_evidence: List[EvidenceDetail] = Field(
        description="A list of specific evidence types identified within the text, relevant to the provided context."
    )
    analysis_summary: Optional[str] = Field(
        None,
        description="A brief summary or commentary on the evidence identification analysis.",
    )


# Nested schema for a measurement type (Agent 4f)
class MeasurementDetail(BaseModel):
    """Represents an identified measurement type."""

    measurement_type: str = Field(
        description=(
            "The classified type of measurement identified (e.g., Financial Metric, Physical Quantity, Performance Indicator, Survey Result, Count, Ratio, Percentage)."
        )
    )
    # Optional: Add unit or value if needed
    # unit: Optional[str] = Field(None, description="The unit of the measurement, if applicable.")
    # value: Optional[str] = Field(None, description="The actual value mentioned, if relevant.")


# Schema for measurement type analysis output (Agent 4f)
class MeasurementTypeSchema(BaseModel):
    """Schema defining the expected output for measurement type analysis (Step 4f)."""

    primary_domain: str = Field(
        description="The primary domain context provided for the analysis."
    )
    analyzed_sub_domains: List[str] = Field(
        description="The list of sub-domains used as context during measurement identification."
    )
    identified_measurements: List[MeasurementDetail] = Field(
        description="A list of specific measurement types identified within the text, relevant to the provided context."
    )
    analysis_summary: Optional[str] = Field(
        None,
        description="A brief summary or commentary on the measurement identification analysis.",
    )


# Nested schema for a modality type (Agent 4g - NEW)
class ModalityDetail(BaseModel):
    """Represents an identified modality type."""

    modality_type: str = Field(
        description=(
            "The classified type of modality identified (e.g., Text, Image, Video, Audio, Table, Chart, Code Snippet, Mathematical Formula)."
        )
    )
    # Optional: Add count or description if needed
    # count: Optional[int] = Field(None, description="Number of times this modality is represented, if applicable.")


# Schema for modality type analysis output (Agent 4g - NEW)
class ModalityTypeSchema(BaseModel):
    """Schema defining the expected output for modality type analysis (Step 4g)."""

    primary_domain: str = Field(
        description="The primary domain context provided for the analysis."
    )
    analyzed_sub_domains: List[str] = Field(
        description="The list of sub-domains used as context during modality identification."
    )
    identified_modalities: List[ModalityDetail] = Field(
        description="A list of specific modality types identified within the text, relevant to the provided context."
    )
    analysis_summary: Optional[str] = Field(
        None,
        description="A brief summary or commentary on the modality identification analysis.",
    )


# --- Schema for Step 5: Entity Instance Extraction ---


class EntityInstanceDetail(BaseModel):
    """Represents a specific entity mention extracted from the text."""

    entity_type: str = Field(
        description="The type of the entity as classified in previous steps."
    )
    text_span: str = Field(description="Exact text of the entity mention.")
    start_char: Optional[int] = Field(
        None,
        description="Start character index of the mention in the full text (0-based).",
    )
    end_char: Optional[int] = Field(
        None,
        description="End character index of the mention in the full text (exclusive).",
    )
    confidence_score: Optional[float] = Field(
        None,
        description="Optional confidence score (0.0 to 1.0) for this entity instance.",
    )
    relevance_score: Optional[float] = Field(
        None,
        description="Optional relevance score (0.0 to 1.0) for this entity instance.",
    )
    clarity_score: Optional[float] = Field(
        None,
        description="Optional clarity score (0.0 to 1.0) for this entity instance.",
    )


class EntityInstanceSchema(BaseModel):
    """Schema defining extracted entity instances within the document."""

    primary_domain: str = Field(
        description="The primary domain context for the extraction."
    )
    analyzed_sub_domains: List[str] = Field(
        description="Sub-domains used as context during extraction."
    )
    analyzed_entity_types: List[str] = Field(
        description="Entity types considered when extracting instances."
    )
    identified_instances: List[EntityInstanceDetail] = Field(
        description="List of extracted entity mentions with type and text span."
    )
    analysis_summary: Optional[str] = Field(
        None, description="Optional summary of the entity instance extraction process."
    )


# --- Schema for Step 5b: Ontology Instance Extraction ---


class OntologyInstanceDetail(BaseModel):
    """Represents a specific ontology concept mention extracted from the text."""

    ontology_type: str = Field(
        description="The ontology type or concept as classified in previous steps."
    )
    text_span: str = Field(description="Exact text of the ontology concept mention.")
    start_char: Optional[int] = Field(
        None,
        description="Start character index of the mention in the full text (0-based).",
    )
    end_char: Optional[int] = Field(
        None,
        description="End character index of the mention in the full text (exclusive).",
    )
    confidence_score: Optional[float] = Field(
        None,
        description="Optional confidence score (0.0 to 1.0) for this ontology instance.",
    )
    relevance_score: Optional[float] = Field(
        None,
        description="Optional relevance score (0.0 to 1.0) for this ontology instance.",
    )
    clarity_score: Optional[float] = Field(
        None,
        description="Optional clarity score (0.0 to 1.0) for this ontology instance.",
    )


class OntologyInstanceSchema(BaseModel):
    """Schema defining extracted ontology instances within the document."""

    primary_domain: str = Field(
        description="The primary domain context for the extraction."
    )
    analyzed_sub_domains: List[str] = Field(
        description="Sub-domains used as context during extraction."
    )
    analyzed_ontology_types: List[str] = Field(
        description="Ontology types considered when extracting instances."
    )
    identified_instances: List[OntologyInstanceDetail] = Field(
        description="List of extracted ontology concept mentions with type and text span."
    )
    analysis_summary: Optional[str] = Field(
        None,
        description="Optional summary of the ontology instance extraction process.",
    )


# --- Schema for Step 5c: Event Instance Extraction ---


class EventInstanceDetail(BaseModel):
    """Represents a specific event mention extracted from the text."""

    event_type: str = Field(
        description="The event type as classified in previous steps."
    )
    text_span: str = Field(description="Exact text of the event mention.")
    start_char: Optional[int] = Field(
        None,
        description="Start character index of the mention in the full text (0-based).",
    )
    end_char: Optional[int] = Field(
        None,
        description="End character index of the mention in the full text (exclusive).",
    )
    confidence_score: Optional[float] = Field(
        None,
        description="Optional confidence score (0.0 to 1.0) for this event instance.",
    )
    relevance_score: Optional[float] = Field(
        None,
        description="Optional relevance score (0.0 to 1.0) for this event instance.",
    )
    clarity_score: Optional[float] = Field(
        None,
        description="Optional clarity score (0.0 to 1.0) for this event instance.",
    )


class EventInstanceSchema(BaseModel):
    """Schema defining extracted event instances within the document."""

    primary_domain: str = Field(
        description="The primary domain context for the extraction."
    )
    analyzed_sub_domains: List[str] = Field(
        description="Sub-domains used as context during extraction."
    )
    analyzed_event_types: List[str] = Field(
        description="Event types considered when extracting instances."
    )
    identified_instances: List[EventInstanceDetail] = Field(
        description="List of extracted event mentions with type and text span."
    )
    analysis_summary: Optional[str] = Field(
        None, description="Optional summary of the event instance extraction process."
    )


# --- Schema for Step 5d: Statement Instance Extraction ---


class StatementInstanceDetail(BaseModel):
    """Represents a specific statement mention extracted from the text."""

    statement_type: str = Field(
        description="The statement type as classified in previous steps."
    )
    text_span: str = Field(
        description="Exact text of the statement or snippet identified."
    )
    start_char: Optional[int] = Field(
        None,
        description="Start character index of the statement in the full text (0-based).",
    )
    end_char: Optional[int] = Field(
        None,
        description="End character index of the statement in the full text (exclusive).",
    )
    confidence_score: Optional[float] = Field(
        None,
        description="Optional confidence score (0.0 to 1.0) for this statement instance.",
    )
    relevance_score: Optional[float] = Field(
        None,
        description="Optional relevance score (0.0 to 1.0) for this statement instance.",
    )
    clarity_score: Optional[float] = Field(
        None,
        description="Optional clarity score (0.0 to 1.0) for this statement instance.",
    )


class StatementInstanceSchema(BaseModel):
    """Schema defining extracted statement instances within the document."""

    primary_domain: str = Field(
        description="The primary domain context for the extraction."
    )
    analyzed_sub_domains: List[str] = Field(
        description="Sub-domains used as context during extraction."
    )
    analyzed_statement_types: List[str] = Field(
        description="Statement types considered when extracting instances."
    )
    identified_instances: List[StatementInstanceDetail] = Field(
        description="List of extracted statement mentions with type and text span."
    )
    analysis_summary: Optional[str] = Field(
        None,
        description="Optional summary of the statement instance extraction process.",
    )


# --- Schema for Step 5e: Evidence Instance Extraction ---


class EvidenceInstanceDetail(BaseModel):
    """Represents a specific evidence mention extracted from the text."""

    evidence_type: str = Field(
        description="The evidence type as classified in previous steps."
    )
    text_span: str = Field(description="Exact text of the evidence mention.")
    start_char: Optional[int] = Field(
        None,
        description="Start character index of the mention in the full text (0-based).",
    )
    end_char: Optional[int] = Field(
        None,
        description="End character index of the mention in the full text (exclusive).",
    )
    confidence_score: Optional[float] = Field(
        None,
        description="Optional confidence score (0.0 to 1.0) for this evidence instance.",
    )
    relevance_score: Optional[float] = Field(
        None,
        description="Optional relevance score (0.0 to 1.0) for this evidence instance.",
    )
    clarity_score: Optional[float] = Field(
        None,
        description="Optional clarity score (0.0 to 1.0) for this evidence instance.",
    )


class EvidenceInstanceSchema(BaseModel):
    """Schema defining extracted evidence instances within the document."""

    primary_domain: str = Field(
        description="The primary domain context for the extraction."
    )
    analyzed_sub_domains: List[str] = Field(
        description="Sub-domains used as context during extraction."
    )
    analyzed_evidence_types: List[str] = Field(
        description="Evidence types considered when extracting instances."
    )
    identified_instances: List[EvidenceInstanceDetail] = Field(
        description="List of extracted evidence mentions with type and text span."
    )
    analysis_summary: Optional[str] = Field(
        None,
        description="Optional summary of the evidence instance extraction process.",
    )


# --- Schema for Step 5f: Measurement Instance Extraction ---


class MeasurementInstanceDetail(BaseModel):
    """Represents a specific measurement mention extracted from the text."""

    measurement_type: str = Field(
        description="The measurement type as classified in previous steps."
    )
    text_span: str = Field(description="Exact text of the measurement mention.")
    start_char: Optional[int] = Field(
        None,
        description="Start character index of the mention in the full text (0-based).",
    )
    end_char: Optional[int] = Field(
        None,
        description="End character index of the mention in the full text (exclusive).",
    )
    confidence_score: Optional[float] = Field(
        None,
        description="Optional confidence score (0.0 to 1.0) for this measurement instance.",
    )
    relevance_score: Optional[float] = Field(
        None,
        description="Optional relevance score (0.0 to 1.0) for this measurement instance.",
    )
    clarity_score: Optional[float] = Field(
        None,
        description="Optional clarity score (0.0 to 1.0) for this measurement instance.",
    )


class MeasurementInstanceSchema(BaseModel):
    """Schema defining extracted measurement instances within the document."""

    primary_domain: str = Field(
        description="The primary domain context for the extraction."
    )
    analyzed_sub_domains: List[str] = Field(
        description="Sub-domains used as context during extraction."
    )
    analyzed_measurement_types: List[str] = Field(
        description="Measurement types considered when extracting instances."
    )
    identified_instances: List[MeasurementInstanceDetail] = Field(
        description="List of extracted measurement mentions with type and text span."
    )
    analysis_summary: Optional[str] = Field(
        None,
        description="Optional summary of the measurement instance extraction process.",
    )


# --- Schema for Step 5g: Modality Instance Extraction ---


class ModalityInstanceDetail(BaseModel):
    """Represents a specific modality mention extracted from the text."""

    modality_type: str = Field(
        description="The modality type as classified in previous steps."
    )
    text_span: str = Field(
        description="Exact text of the modality mention or reference."
    )
    start_char: Optional[int] = Field(
        None,
        description="Start character index of the mention in the full text (0-based).",
    )
    end_char: Optional[int] = Field(
        None,
        description="End character index of the mention in the full text (exclusive).",
    )
    confidence_score: Optional[float] = Field(
        None,
        description="Optional confidence score (0.0 to 1.0) for this modality instance.",
    )
    relevance_score: Optional[float] = Field(
        None,
        description="Optional relevance score (0.0 to 1.0) for this modality instance.",
    )
    clarity_score: Optional[float] = Field(
        None,
        description="Optional clarity score (0.0 to 1.0) for this modality instance.",
    )


class ModalityInstanceSchema(BaseModel):
    """Schema defining extracted modality instances within the document."""

    primary_domain: str = Field(
        description="The primary domain context for the extraction."
    )
    analyzed_sub_domains: List[str] = Field(
        description="Sub-domains used as context during extraction."
    )
    analyzed_modality_types: List[str] = Field(
        description="Modality types considered when extracting instances."
    )
    identified_instances: List[ModalityInstanceDetail] = Field(
        description="List of extracted modality mentions with type and text span."
    )
    analysis_summary: Optional[str] = Field(
        None,
        description="Optional summary of the modality instance extraction process.",
    )


# --- Schemas for Step 5 (Relationship Identification) ---


# Nested schema for a specific identified relationship between entities
class RelationshipDetail(BaseModel):
    """Represents a single identified relationship between two entities."""

    relationship_type: str = Field(
        description="The nature of the relationship identified (e.g., WORKS_FOR, LOCATED_IN, ACQUIRED, PARTNERS_WITH, COMPETES_WITH)."
    )
    relevance_score: Optional[float] = Field(
        None,
        description="Optional relevance score (0.0 to 1.0) for identifying this relationship type.",
    )
    # Optional: Add a field for the sentence/snippet supporting the relationship
    # supporting_text: Optional[str] = Field(None, description="The text snippet that supports this relationship finding.")


# Schema for the output of a *single* relationship identification agent call (Agent 5 - one call per entity type focus)
class SingleEntityTypeRelationshipSchema(BaseModel):
    """Represents relationships identified focusing on a single entity type within the broader context."""

    entity_type_focus: str = Field(
        description="The specific entity type this analysis focused on (e.g., ORGANIZATION)."
    )
    identified_relationships: List[RelationshipDetail] = Field(
        description="A list of relationships involving the focus entity type identified within the text, relevant to the overall context."
    )


# Schema for the final, aggregated relationship output (used for saving Step 5 results)
class RelationshipSchema(BaseModel):
    """Schema defining the final aggregated output for relationship identification analysis."""

    primary_domain: str = Field(
        description="The overall primary domain provided as context."
    )
    analyzed_sub_domains: List[str] = Field(
        description="The list of sub-domains used as context during relationship identification."
    )
    analyzed_entity_types: List[str] = Field(
        description="The list of entity types that were analyzed in parallel for relationships."
    )
    # Contains the results from each parallel call, one for each analyzed entity type
    entity_relationships_map: List[SingleEntityTypeRelationshipSchema] = Field(
        description="A list containing the relationship analysis results for each entity type focus."
    )
    analysis_summary: Optional[str] = Field(
        None,
        description="Optional summary of the relationship identification process across all analyzed entity types.",
    )


# --- Schema for Step 6: Relationship Instance Extraction ---
class RelationshipInstanceDetail(BaseModel):
    """Represents a specific relationship instance between two entities."""

    subject: str = Field(
        description="The text span or identifier of the subject entity."
    )
    relationship_type: str = Field(
        description="The type of relationship linking the subject and object."
    )
    object: str = Field(description="The text span or identifier of the object entity.")
    snippet: Optional[str] = Field(
        None, description="Optional text snippet supporting this relationship instance."
    )
    confidence_score: Optional[float] = Field(
        None,
        description="Optional confidence score (0.0 to 1.0) for this relationship instance.",
    )
    relevance_score: Optional[float] = Field(
        None,
        description="Optional relevance score (0.0 to 1.0) for this relationship instance.",
    )
    clarity_score: Optional[float] = Field(
        None,
        description="Optional clarity score (0.0 to 1.0) for this relationship instance.",
    )


class RelationshipInstanceSchema(BaseModel):
    """Schema defining extracted relationship instances within the document."""

    primary_domain: str = Field(
        description="The primary domain context for the extraction."
    )
    analyzed_sub_domains: List[str] = Field(
        description="Sub-domains used as context during extraction."
    )
    identified_instances: List[RelationshipInstanceDetail] = Field(
        description="List of extracted relationship instances between specific entities."
    )
    analysis_summary: Optional[str] = Field(
        None,
        description="Optional summary of the relationship instance extraction process.",
    )


# --- Aggregated Extracted Instances Schema ---
class ExtractedInstancesSchema(BaseModel):
    """Aggregates all instance extraction outputs from Step 5."""

    primary_domain: str = Field(
        description="Overall primary domain context for the extracted instances."
    )
    analyzed_sub_domains: List[str] = Field(
        description="Sub-domains considered during instance extraction."
    )

    entity_instances: List[EntityInstanceDetail] = Field(
        default_factory=list,
        description="List of extracted entity mentions across the document.",
    )
    ontology_instances: List[OntologyInstanceDetail] = Field(
        default_factory=list, description="List of extracted ontology concept mentions."
    )
    event_instances: List[EventInstanceDetail] = Field(
        default_factory=list, description="List of extracted event mentions."
    )
    statement_instances: List[StatementInstanceDetail] = Field(
        default_factory=list, description="List of extracted statement mentions."
    )
    evidence_instances: List[EvidenceInstanceDetail] = Field(
        default_factory=list, description="List of extracted evidence mentions."
    )
    measurement_instances: List[MeasurementInstanceDetail] = Field(
        default_factory=list, description="List of extracted measurement mentions."
    )
    modality_instances: List[ModalityInstanceDetail] = Field(
        default_factory=list, description="List of extracted modality mentions."
    )
    relationship_instances: List[RelationshipInstanceDetail] = Field(
        default_factory=list, description="List of extracted relationship instances."
    )
    analysis_summary: Optional[str] = Field(
        None, description="Optional summary describing the aggregated instance data."
    )
