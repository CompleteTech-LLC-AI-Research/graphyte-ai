# File: /Users/completetech/Desktop/python-agent-sdk/src/agentic_team_workflow/agents.py
# NOTE: Using 'agentic_team' as the alias for the SDK import
try:
    # Assuming 'agents' is the correct SDK import name based on previous examples
    # If your alias is truly 'agentic_team', adjust the import accordingly.
    from agents import Agent
except ImportError:
    print("Error: Agent SDK library not found or incomplete. Cannot define agents.")
    # Depending on execution context, might want `sys.exit(1)` here,
    # but typically module-level errors are handled by the importer.
    raise # Re-raise the import error

from .schemas import (
    DomainSchema, SubDomainSchema, SingleSubDomainTopicSchema,
    EntityTypeSchema, OntologyTypeSchema, EventSchema,
    StatementTypeSchema, EvidenceTypeSchema, MeasurementTypeSchema,
    ModalityTypeSchema, SingleEntityTypeRelationshipSchema
)
from .config import (
    DOMAIN_MODEL, SUB_DOMAIN_MODEL, TOPIC_MODEL,
    ENTITY_TYPE_MODEL, ONTOLOGY_TYPE_MODEL, EVENT_TYPE_MODEL,
    STATEMENT_TYPE_MODEL, EVIDENCE_TYPE_MODEL, MEASUREMENT_TYPE_MODEL,
    MODALITY_TYPE_MODEL, RELATIONSHIP_MODEL
)

# --- Agent 1: Domain Identifier ---
domain_identifier_agent = Agent(
    name="DomainIdentifierAgent",
    instructions=(
        "Your primary task: Analyze the provided text content and identify the single, most relevant high-level domain. "
        "Examples include: Finance, Technology, Healthcare, Arts, Science, Entertainment, Sports, "
        "Politics, Education, Environment, Business, Lifestyle, Travel, etc. "
        "Focus on the *primary* topic.\n"
        "Also provide a confidence score between 0.0 (uncertain) and 1.0 (very confident) that the identified domain is the correct primary domain for the entire text. "
        "Output ONLY the result using the provided DomainSchema, including the domain and the confidence_score. Do not add any other commentary."
    ),
    model=DOMAIN_MODEL,
    output_type=DomainSchema,
    tools=[],
    handoffs=[],
)

# --- Agent 2: Sub-Domain Identifier ---
sub_domain_identifier_agent = Agent(
    name="SubDomainIdentifierAgent",
    instructions=(
        "You are given text content and its primary domain. Your task is to identify specific sub-domains "
        "within the text related to the primary domain. "
        "For EACH identified sub-domain, provide a relevance score between 0.0 (not relevant) and 1.0 (highly relevant) indicating how strongly it relates to the primary domain within the context of the text. "
        "Also provide a brief overall analysis summary.\n"
        "Output ONLY the result using the provided SubDomainSchema. Ensure the identified_sub_domains field contains a list of items, each with a sub_domain string and a relevance_score."
    ),
    model=SUB_DOMAIN_MODEL,
    tools=[],
    handoffs=[],
    output_type=SubDomainSchema,
)

# --- Agent 3: Topic Identifier (for one sub-domain) ---
topic_identifier_agent = Agent(
    name="TopicIdentifierAgent",
    instructions=(
        "You are provided with text, its primary domain, and ONE specific sub-domain. "
        "Your task: Analyze the *full text* and identify specific, relevant topics mentioned within the text that fall under the provided single sub-domain. "
        "For EACH identified topic, provide a relevance score between 0.0 (not relevant) and 1.0 (highly relevant) indicating how strongly the topic relates to the specified sub-domain within the context of the text. "
        "Output the results ONLY using the provided SingleSubDomainTopicSchema, including the topic string and its relevance_score for each item in the identified_topics list."
    ),
    model=TOPIC_MODEL,
    tools=[],
    handoffs=[],
    output_type=SingleSubDomainTopicSchema,
)


# --- Base Agent for Type Identification (Agents 4a-4g) ---
# This base agent provides a template for identifying various concept types.
# It will be cloned and specialized for each specific type.

base_type_identifier_instructions_template = (
    "Your primary task: Analyze the provided text content to identify key {concept_description}. {specific_constraint} "
    "You will be given the full text AND context about its primary domain, identified sub-domains, and relevant topics found in previous analysis steps.\n"
    "Use this context to assess the relevance of each identified {concept_type_singular} to the overall subject matter.\n"
    "For EACH identified {concept_type_singular}, provide:\n"
    "1. The classified {concept_type_singular}.\n"
    "2. A relevance score between 0.0 (not relevant to the context) and 1.0 (highly relevant to the context).\n"
    "Provide an overall analysis summary if applicable.\n"
    "Output ONLY the result using the provided schema structure. Ensure the {list_field_name} field contains a list of items, each with '{item_field_name}' and 'relevance_score'. Include the 'primary_domain' and 'analyzed_sub_domains' fields from the context in your output schema."
)

base_type_identifier_agent = Agent(
    name="BaseTypeIdentifierAgent", # Generic name, will be overridden
    instructions=base_type_identifier_instructions_template, # Will be formatted in clones
    # No default model or output_type, must be specified in clones
    tools=[],
    handoffs=[],
)


# --- Agent 4a: Entity Type Identifier ---
entity_type_identifier_agent = base_type_identifier_agent.clone(
    name="EntityTypeIdentifierAgent",
    instructions=base_type_identifier_instructions_template.format(
        concept_description="entity types (e.g., PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PRODUCT, TECHNOLOGY, SCIENTIFIC_CONCEPT, ECONOMIC_INDICATOR)",
        specific_constraint="Do NOT identify Event types - that is handled by another agent.", # Retain original constraint
        concept_type_singular="entity type",
        list_field_name="identified_entities",
        item_field_name="entity_type"
    ),
    model=ENTITY_TYPE_MODEL,
    output_type=EntityTypeSchema,
)

# --- Agent 4b: Ontology Type Identifier ---
ontology_type_identifier_agent = base_type_identifier_agent.clone(
    name="OntologyTypeIdentifierAgent",
    instructions=base_type_identifier_instructions_template.format(
        concept_description="relevant ontology types or concepts, potentially referencing standard ontologies (like Schema.org, FIBO, domain-specific ones) where applicable",
        specific_constraint="Focus on conceptual or taxonomic classifications, potentially referencing standard ontologies (like Schema.org, FIBO). Avoid simple entity labels.", # Added constraint
        concept_type_singular="ontology type/concept",
        list_field_name="identified_ontology_types",
        item_field_name="ontology_type"
    ),
    model=ONTOLOGY_TYPE_MODEL,
    output_type=OntologyTypeSchema,
)

# --- Agent 4c: Event Type Identifier ---
event_type_identifier_agent = base_type_identifier_agent.clone(
    name="EventTypeIdentifierAgent",
    instructions=base_type_identifier_instructions_template.format(
        concept_description="key EVENT types (e.g., Meeting, Acquisition, Conference, Product Launch, Election, Natural Disaster, Release, Protest, Accident, Celebration)",
        specific_constraint="Do NOT identify other entity types like Person, Organization, Location etc. - focus ONLY on events.", # Retain original constraint
        concept_type_singular="event type",
        list_field_name="identified_events",
        item_field_name="event_type"
    ),
    model=EVENT_TYPE_MODEL,
    output_type=EventSchema,
)

# --- Agent 4d: Statement Type Identifier ---
statement_type_identifier_agent = base_type_identifier_agent.clone(
    name="StatementTypeIdentifierAgent",
    instructions=base_type_identifier_instructions_template.format(
        concept_description="key STATEMENT types (e.g., Fact, Claim, Opinion, Question, Instruction, Hypothesis, Prediction)",
        specific_constraint="Focus only on classifying the nature or type of the statement (e.g., Fact, Opinion, Claim, Hypothesis), not its specific content or truth value.", # Added constraint
        concept_type_singular="statement type",
        list_field_name="identified_statements",
        item_field_name="statement_type"
    ),
    model=STATEMENT_TYPE_MODEL,
    output_type=StatementTypeSchema,
)

# --- Agent 4e: Evidence Type Identifier ---
evidence_type_identifier_agent = base_type_identifier_agent.clone(
    name="EvidenceTypeIdentifierAgent",
    instructions=base_type_identifier_instructions_template.format(
        concept_description="key types of EVIDENCE presented (e.g., Testimony, Document Reference, Statistic, Anecdote, Expert Opinion, Observation, Example, Case Study, Logical Argument)",
        specific_constraint="Focus on the *form* or *category* of evidence used to support claims or statements (e.g., Statistic, Testimony, Document Reference).", # Added constraint
        concept_type_singular="evidence type",
        list_field_name="identified_evidence",
        item_field_name="evidence_type"
    ),
    model=EVIDENCE_TYPE_MODEL,
    output_type=EvidenceTypeSchema,
)

# --- Agent 4f: Measurement Type Identifier ---
measurement_type_identifier_agent = base_type_identifier_agent.clone(
    name="MeasurementTypeIdentifierAgent",
    instructions=base_type_identifier_instructions_template.format(
        concept_description="key types of MEASUREMENTS mentioned (e.g., Financial Metric, Physical Quantity, Performance Indicator, Survey Result, Count, Ratio, Percentage, Score)",
        specific_constraint="Focus on the *category* or *type* of measurement being used (e.g., Financial Metric, Physical Quantity, Ratio), not necessarily the specific values.", # Added constraint
        concept_type_singular="measurement type",
        list_field_name="identified_measurements",
        item_field_name="measurement_type"
    ),
    model=MEASUREMENT_TYPE_MODEL,
    output_type=MeasurementTypeSchema,
)

# --- Agent 4g: Modality Type Identifier ---
modality_type_identifier_agent = base_type_identifier_agent.clone(
    name="ModalityTypeIdentifierAgent",
    instructions=base_type_identifier_instructions_template.format(
        concept_description="the types of MODALITIES represented or referred to (e.g., Text, Image, Video, Audio, Table, Chart, Code Snippet, Mathematical Formula, Diagram)",
        specific_constraint="Identify the *format or medium* of information presented or referenced (e.g., Text, Image, Table, Code Snippet).", # Added constraint
        concept_type_singular="modality type",
        list_field_name="identified_modalities",
        item_field_name="modality_type"
    ),
    model=MODALITY_TYPE_MODEL,
    output_type=ModalityTypeSchema,
)


# --- Agent 5: Relationship Identifier (for one entity type) ---
relationship_type_identifier_agent = Agent(
    name="RelationshipTypeIdentifierAgent",
    instructions=(
        "Your task: Analyze the provided text and context (domain, sub-domains, topics, and entity types) to identify relationships between entities. "
        "You will be given the *full text* and context, PLUS **one specific entity type** (identified in a previous step) to focus on (e.g., 'ORGANIZATION').\n"
        "Identify explicit or strongly implied relationships mentioned in the text where the **focus entity type** is involved as one of the participants.\n"
        "Examples of relationships: WORKS_FOR, LOCATED_IN, ACQUIRED, PARTNERED_WITH, COMPETES_WITH, FOUNDED_BY, MANUFACTURES, USES_TECHNOLOGY, etc.\n"
        "For EACH identified relationship involving the focus entity type:\n"
        "1. State the unique type of relationship found.\n"
        "2. Provide a relevance score (0.0 to 1.0) indicating the confidence/clarity of this relationship based on the text and context.\n"
        "Output ONLY the result using the provided SingleEntityTypeRelationshipSchema. Ensure the 'entity_type_focus' field matches the entity type you were asked to focus on, and populate the 'identified_relationships' list with RelationshipDetail objects. Do not add commentary outside the schema."
    ),
    model=RELATIONSHIP_MODEL,
    output_type=SingleEntityTypeRelationshipSchema,
    tools=[],
    handoffs=[],
)

# You can optionally create a list or dict to easily access all agents
all_agents = {
    "domain_identifier": domain_identifier_agent,
    "sub_domain_identifier": sub_domain_identifier_agent,
    "topic_identifier": topic_identifier_agent,
    "entity_type_identifier": entity_type_identifier_agent,
    "ontology_type_identifier": ontology_type_identifier_agent,
    "event_type_identifier": event_type_identifier_agent,
    "statement_type_identifier": statement_type_identifier_agent,
    "evidence_type_identifier": evidence_type_identifier_agent,
    "measurement_type_identifier": measurement_type_identifier_agent,
    "modality_type_identifier": modality_type_identifier_agent,
    "relationship_identifier": relationship_type_identifier_agent,
    # Note: Base agent is not typically included here unless used directly
}