# File: /Users/completetech/Desktop/python-agent-sdk/src/agentic_team_workflow/schemas.py
from typing import List, Optional
from pydantic import BaseModel, Field

# --- Schemas for Existing Agents (1-3) ---

# Schema for primary domain output (Agent 1)
class DomainSchema(BaseModel):
    """Schema defining the expected output: the primary domain and confidence."""
    domain: str = Field(description=(
        "The single, most relevant high-level domain identified in the text "
        "(e.g., Technology, Finance, Healthcare, Arts, Science, Entertainment, Sports, Politics)."
    ))
    confidence_score: float = Field(
        description="The confidence level (0.0 to 1.0) that the identified domain is the correct primary domain for the entire text."
    )

# Nested schema for a sub-domain and its relevance score
class SubDomainWithScore(BaseModel):
    """Represents a single identified sub-domain and its relevance score."""
    sub_domain: str = Field(description="The specific sub-domain identified within the text.")
    relevance_score: float = Field(
        description="The relevance of this sub-domain to the primary domain, as a score between 0.0 (not relevant) and 1.0 (highly relevant)."
    )

# Schema for sub-domain analysis output (Agent 2)
class SubDomainSchema(BaseModel):
    """Schema defining the expected output for sub-domain analysis, including relevance scores."""
    primary_domain: str = Field(description="The primary domain that was provided as input.")
    identified_sub_domains: List[SubDomainWithScore] = Field(
        description="A list of specific sub-domains identified within the text related to the primary domain, each with a relevance score. Should not be empty if analysis was possible."
    )
    analysis_summary: Optional[str] = Field(None, description="A brief summary or commentary on the sub-domain analysis.")

# Nested schema for a topic and its relevance score
class TopicWithScore(BaseModel):
    """Represents a single identified topic and its relevance score."""
    topic: str = Field(description="The specific topic identified within the text.")
    relevance_score: float = Field(
        description="The relevance of this topic to the specified sub-domain, as a score between 0.0 (not relevant) and 1.0 (highly relevant)."
    )

# Schema for the output of the *single* topic identification agent call (Agent 3)
class SingleSubDomainTopicSchema(BaseModel):
    """Represents topics and their relevance scores identified for a single sub-domain."""
    sub_domain: str = Field(description="The sub-domain being analyzed.")
    identified_topics: List[TopicWithScore] = Field(
        description="A list of specific topics identified within the text related to this sub-domain, each with a relevance score."
    )

# Schema for the final, aggregated topic output (used for saving)
class TopicSchema(BaseModel):
    """Schema defining the final aggregated output for topic identification analysis."""
    primary_domain: str = Field(description="The overall primary domain provided.")
    sub_domain_topic_map: List[SingleSubDomainTopicSchema] = Field(description="A list mapping each analyzed sub-domain to its identified topics and their scores.")
    analysis_summary: Optional[str] = Field(None, description="Optional summary of the topic identification process across all sub-domains (can be generated or left null).")


# --- Schemas for Step 4 Agents (4a, 4b, 4c, 4d, 4e, 4f, 4g) ---

# Nested schema for an entity type and relevance score (Agent 4a)
class EntityWithTypeScore(BaseModel):
    """Represents an entity type and its relevance score."""
    entity_type: str = Field(description=(
        "The classified type of the entity (e.g., PERSON, ORGANIZATION, LOCATION, "
        "DATE, MONEY, PRODUCT, TECHNOLOGY, SCIENTIFIC_CONCEPT, ECONOMIC_INDICATOR)." # Removed EVENT as it's now separate
    ))
    relevance_score: float = Field(
        description="The relevance score (0.0 to 1.0) of this entity type to the overall context "
                    "(document, domain, sub-domains, topics)."
    )

# Schema for entity type analysis output (Agent 4a)
class EntityTypeSchema(BaseModel):
    """Schema defining the expected output for entity type analysis (Step 4a)."""
    primary_domain: str = Field(description="The primary domain context provided for the analysis.")
    analyzed_sub_domains: List[str] = Field(
        description="The list of sub-domains used as context during entity identification."
    )
    identified_entities: List[EntityWithTypeScore] = Field(
        description="A list of specific entity types identified within the text, relevant to the provided context, each with a relevance score."
    )
    analysis_summary: Optional[str] = Field(None, description="A brief summary or commentary on the entity identification analysis.")


# Nested schema for an ontology type/concept and relevance score (Agent 4b)
class OntologyTypeWithScore(BaseModel):
    """Represents an ontology type/concept and its relevance score."""
    ontology_type: str = Field(description=(
        "The identified ontology type or concept (e.g., Schema.org:Person, FIBO:FinancialInstrument, GO:biological_process)."
    ))
    relevance_score: float = Field(
        description="The relevance score (0.0 to 1.0) of this ontology type to the overall context "
                    "(document, domain, sub-domains, topics)."
    )

# Schema for ontology type analysis output (Agent 4b)
class OntologyTypeSchema(BaseModel):
    """Schema defining the expected output for ontology type analysis (Step 4b)."""
    primary_domain: str = Field(description="The primary domain context provided for the analysis.")
    analyzed_sub_domains: List[str] = Field(
        description="The list of sub-domains used as context during ontology identification."
    )
    identified_ontology_types: List[OntologyTypeWithScore] = Field(
        description="A list of specific ontology types/concepts identified within the text, relevant to the provided context, each with a relevance score."
    )
    analysis_summary: Optional[str] = Field(None, description="A brief summary or commentary on the ontology identification analysis.")


# Nested schema for an event type and relevance score (Agent 4c)
class EventDetail(BaseModel):
    """Represents an identified event type and its relevance score."""
    event_type: str = Field(description=(
        "The classified type of the event identified (e.g., Meeting, Acquisition, Conference, Product Launch, Election, Natural Disaster)."
    ))
    # Optional: Add description if needed, like 'Brief description of the event instance'
    # description: Optional[str] = Field(None, description="A brief description or name of the specific event instance.")
    relevance_score: float = Field(
        description="The relevance score (0.0 to 1.0) of this event type to the overall context "
                    "(document, domain, sub-domains, topics)."
    )

# Schema for event type analysis output (Agent 4c)
class EventSchema(BaseModel):
    """Schema defining the expected output for event type analysis (Step 4c)."""
    primary_domain: str = Field(description="The primary domain context provided for the analysis.")
    analyzed_sub_domains: List[str] = Field(
        description="The list of sub-domains used as context during event identification."
    )
    identified_events: List[EventDetail] = Field(
        description="A list of specific event types identified within the text, relevant to the provided context, each with a relevance score."
    )
    analysis_summary: Optional[str] = Field(None, description="A brief summary or commentary on the event identification analysis.")


# Nested schema for a specific event mention extracted from the text (Step 5)
class EventInstanceDetail(BaseModel):
    """Represents a single event mention along with its classified type."""
    event_type: str = Field(description="The event type this mention represents (e.g., Meeting, Acquisition).")
    mention_text: str = Field(description="The text snippet referencing this specific event instance.")
    relevance_score: float = Field(
        description="Confidence score (0.0 to 1.0) that this text truly represents the specified event type."
    )


# Schema for event instance extraction output (used in Step 5)
class EventInstanceSchema(BaseModel):
    """Schema defining extracted event mentions from the document."""
    primary_domain: str = Field(description="The primary domain context provided for the extraction.")
    analyzed_sub_domains: List[str] = Field(
        description="The list of sub-domains used as context during event extraction."
    )
    event_instances: List[EventInstanceDetail] = Field(
        description="List of event instances mentioned in the text with their classified types and relevance scores."
    )
    analysis_summary: Optional[str] = Field(None, description="Optional summary of the event instance extraction process.")


# Nested schema for a statement type and relevance score (Agent 4d)
class StatementDetail(BaseModel):
    """Represents an identified statement type and its relevance score."""
    statement_type: str = Field(description=(
        "The classified type of the statement identified (e.g., Fact, Claim, Opinion, Question, Instruction, Hypothesis, Prediction)."
    ))
    # Optional: Add a snippet of the text classified
    # supporting_text: Optional[str] = Field(None, description="The text snippet classified as this statement type.")
    relevance_score: float = Field(
        description="The relevance score (0.0 to 1.0) of this statement type to the overall context "
                    "(document, domain, sub-domains, topics)."
    )

# Schema for statement type analysis output (Agent 4d)
class StatementTypeSchema(BaseModel):
    """Schema defining the expected output for statement type analysis (Step 4d)."""
    primary_domain: str = Field(description="The primary domain context provided for the analysis.")
    analyzed_sub_domains: List[str] = Field(
        description="The list of sub-domains used as context during statement identification."
    )
    identified_statements: List[StatementDetail] = Field(
        description="A list of specific statement types identified within the text, relevant to the provided context, each with a relevance score."
    )
    analysis_summary: Optional[str] = Field(None, description="A brief summary or commentary on the statement identification analysis.")


# Nested schema for an evidence type and relevance score (Agent 4e)
class EvidenceDetail(BaseModel):
    """Represents an identified evidence type and its relevance score."""
    evidence_type: str = Field(description=(
        "The classified type of evidence identified (e.g., Testimony, Document, Statistic, Anecdote, Expert Opinion, Observation, Example)."
    ))
    relevance_score: float = Field(
        description="The relevance score (0.0 to 1.0) of this evidence type to the overall context "
                    "(document, domain, sub-domains, topics)."
    )

# Schema for evidence type analysis output (Agent 4e)
class EvidenceTypeSchema(BaseModel):
    """Schema defining the expected output for evidence type analysis (Step 4e)."""
    primary_domain: str = Field(description="The primary domain context provided for the analysis.")
    analyzed_sub_domains: List[str] = Field(
        description="The list of sub-domains used as context during evidence identification."
    )
    identified_evidence: List[EvidenceDetail] = Field(
        description="A list of specific evidence types identified within the text, relevant to the provided context, each with a relevance score."
    )
    analysis_summary: Optional[str] = Field(None, description="A brief summary or commentary on the evidence identification analysis.")


# Nested schema for a measurement type and relevance score (Agent 4f)
class MeasurementDetail(BaseModel):
    """Represents an identified measurement type and its relevance score."""
    measurement_type: str = Field(description=(
        "The classified type of measurement identified (e.g., Financial Metric, Physical Quantity, Performance Indicator, Survey Result, Count, Ratio, Percentage)."
    ))
    # Optional: Add unit or value if needed
    # unit: Optional[str] = Field(None, description="The unit of the measurement, if applicable.")
    # value: Optional[str] = Field(None, description="The actual value mentioned, if relevant.")
    relevance_score: float = Field(
        description="The relevance score (0.0 to 1.0) of this measurement type to the overall context "
                    "(document, domain, sub-domains, topics)."
    )

# Schema for measurement type analysis output (Agent 4f)
class MeasurementTypeSchema(BaseModel):
    """Schema defining the expected output for measurement type analysis (Step 4f)."""
    primary_domain: str = Field(description="The primary domain context provided for the analysis.")
    analyzed_sub_domains: List[str] = Field(
        description="The list of sub-domains used as context during measurement identification."
    )
    identified_measurements: List[MeasurementDetail] = Field(
        description="A list of specific measurement types identified within the text, relevant to the provided context, each with a relevance score."
    )
    analysis_summary: Optional[str] = Field(None, description="A brief summary or commentary on the measurement identification analysis.")


# Nested schema for a modality type and relevance score (Agent 4g - NEW)
class ModalityDetail(BaseModel):
    """Represents an identified modality type and its relevance score."""
    modality_type: str = Field(description=(
        "The classified type of modality identified (e.g., Text, Image, Video, Audio, Table, Chart, Code Snippet, Mathematical Formula)."
    ))
    # Optional: Add count or description if needed
    # count: Optional[int] = Field(None, description="Number of times this modality is represented, if applicable.")
    relevance_score: float = Field(
        description="The relevance score (0.0 to 1.0) of this modality type to the overall context "
                    "(document, domain, sub-domains, topics)."
    )

# Schema for modality type analysis output (Agent 4g - NEW)
class ModalityTypeSchema(BaseModel):
    """Schema defining the expected output for modality type analysis (Step 4g)."""
    primary_domain: str = Field(description="The primary domain context provided for the analysis.")
    analyzed_sub_domains: List[str] = Field(
        description="The list of sub-domains used as context during modality identification."
    )
    identified_modalities: List[ModalityDetail] = Field(
        description="A list of specific modality types identified within the text, relevant to the provided context, each with a relevance score."
    )
    analysis_summary: Optional[str] = Field(None, description="A brief summary or commentary on the modality identification analysis.")


# --- Schemas for Step 5 (Relationship Identification) ---

# Nested schema for a specific identified relationship between entities
class RelationshipDetail(BaseModel):
    """Represents a single identified relationship between two entities."""
    relationship_type: str = Field(description="The nature of the relationship identified (e.g., WORKS_FOR, LOCATED_IN, ACQUIRED, PARTNERS_WITH, COMPETES_WITH).")
    relevance_score: float = Field(
        description="The relevance or confidence score (0.0 to 1.0) for this specific relationship based on the text and context."
    )
    # Optional: Add a field for the sentence/snippet supporting the relationship
    # supporting_text: Optional[str] = Field(None, description="The text snippet that supports this relationship finding.")

# Schema for the output of a *single* relationship identification agent call (Agent 5 - one call per entity type focus)
class SingleEntityTypeRelationshipSchema(BaseModel):
    """Represents relationships identified focusing on a single entity type within the broader context."""
    entity_type_focus: str = Field(description="The specific entity type this analysis focused on (e.g., ORGANIZATION).")
    identified_relationships: List[RelationshipDetail] = Field(
        description="A list of relationships involving the focus entity type identified within the text, relevant to the overall context."
    )

# Schema for the final, aggregated relationship output (used for saving Step 5 results)
class RelationshipSchema(BaseModel):
    """Schema defining the final aggregated output for relationship identification analysis."""
    primary_domain: str = Field(description="The overall primary domain provided as context.")
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
    analysis_summary: Optional[str] = Field(None, description="Optional summary of the relationship identification process across all analyzed entity types.")