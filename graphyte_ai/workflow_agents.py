# NOTE: Using the external ``agents`` SDK for agent definitions
from typing import Any, cast, List

try:
    from agents import Agent  # type: ignore[attr-defined]
except ImportError:
    print("Error: 'agents' SDK library not found or incomplete. Cannot define agents.")
    # Depending on execution context, might want `sys.exit(1)` here,
    # but typically module-level errors are handled by the importer.
    Agent = Any  # type: ignore[misc]

from .schemas import (
    DomainSchema,
    DomainResultSchema,
    SubDomainIdentifierSchema,
    SubDomainSchema,
    SingleSubDomainTopicSchema,
    TopicSchema,
    EntityTypeIdentifierSchema,
    OntologyTypeIdentifierSchema,
    EventTypeIdentifierSchema,
    StatementTypeIdentifierSchema,
    EvidenceTypeIdentifierSchema,
    MeasurementTypeIdentifierSchema,
    ModalityTypeIdentifierSchema,
    EntityInstanceSchema,
    StatementInstanceSchema,
    EvidenceInstanceSchema,
    MeasurementInstanceSchema,
    ModalityInstanceSchema,
    SingleEntityTypeRelationshipSchema,
    RelationshipInstanceSchema,
    OntologyInstanceSchema,
    EventInstanceSchema,
    ConfidenceScoreSchema,
    RelevanceScoreSchema,
    ClarityScoreSchema,
)
from .config import (
    DOMAIN_MODEL,
    SUB_DOMAIN_MODEL,
    TOPIC_MODEL,
    ENTITY_TYPE_MODEL,
    ONTOLOGY_TYPE_MODEL,
    EVENT_TYPE_MODEL,
    STATEMENT_TYPE_MODEL,
    EVIDENCE_TYPE_MODEL,
    MEASUREMENT_TYPE_MODEL,
    MODALITY_TYPE_MODEL,
    ENTITY_INSTANCE_MODEL,
    ONTOLOGY_INSTANCE_MODEL,
    EVENT_INSTANCE_MODEL,
    STATEMENT_INSTANCE_MODEL,
    EVIDENCE_INSTANCE_MODEL,
    MEASUREMENT_INSTANCE_MODEL,
    MODALITY_INSTANCE_MODEL,
    RELATIONSHIP_MODEL,
    RELATIONSHIP_INSTANCE_MODEL,
    DEFAULT_MODEL,
)

# --- Base Agent for Scoring ---
# Template agent used for calculating confidence or relevance scores.
base_scoring_instructions_template = (
    "Evaluate the provided {item_description} and assign a numeric {score_type} between 0.0 and 1.0. "
    "Use any available context to inform your assessment. "
    "Output ONLY JSON using the schema with the '{score_field}' field."
)

base_scoring_agent = Agent(
    name="BaseScoringAgent",  # Generic name, overridden in clones
    instructions=base_scoring_instructions_template,
    tools=[],
    handoffs=[],
)

# --- Confidence Score Agent ---
# Specialized clone of the base scoring agent used to assess
# confidence in a domain classification or relationship instance.
confidence_score_agent = base_scoring_agent.clone(
    name="ConfidenceScoreAgent",
    instructions=base_scoring_instructions_template.format(
        item_description="domain or relationship instance",
        score_type="confidence score ",
        score_field="confidence_score",
    ),
    model=DEFAULT_MODEL,
    output_type=ConfidenceScoreSchema,
)

# --- Relevance Score Agent ---
# Clone of the base scoring agent used to judge relevance of items like
# sub-domains, topics, types, or relationship types.
relevance_score_agent = base_scoring_agent.clone(
    name="RelevanceScoreAgent",
    instructions=base_scoring_instructions_template.format(
        item_description=(
            "sub-domain, topic, entity/ontology/event/statement/evidence/"
            "measurement/modality type, or relationship type"
        ),
        score_type="relevance score ",
        score_field="relevance_score",
    ),
    model=DEFAULT_MODEL,
    output_type=RelevanceScoreSchema,
)

# --- Clarity Score Agent ---
# Clone of the base scoring agent used to assess clarity of text, relationships, or entities.
clarity_score_agent = base_scoring_agent.clone(
    name="ClarityScoreAgent",
    instructions=base_scoring_instructions_template.format(
        item_description="text, relationship, or entity",
        score_type="clarity score ",
        score_field="clarity_score",
    ),
    model=DEFAULT_MODEL,
    output_type=ClarityScoreSchema,
)

# --- Result Agent Helper ---
# Allows cloning an existing agent to simply return a provided item
# along with pre-calculated scores.
result_agent_instructions_template = (
    "You are provided with a {item_description} and pre-calculated confidence, "
    "relevance, and clarity scores. Do not recompute these values. "
    "Return ONLY valid JSON conforming to {schema_name} that "
    "includes the supplied {item_description} and scores."
)


def create_result_agent(base_agent: Agent, schema: Any, item_description: str) -> Agent:
    """Clone ``base_agent`` and configure it to return a result with scores.

    Parameters
    ----------
    base_agent : Agent
        The agent instance to clone.
    schema : Any
        Output schema that the result should conform to.
    item_description : str
        Description of the item being returned (e.g., "domain label").

    Returns
    -------
    Agent
        A new agent that echoes the provided result and scores.
    """

    return base_agent.clone(
        name=base_agent.name.replace("Identifier", "Result"),
        instructions=result_agent_instructions_template.format(
            item_description=item_description,
            schema_name=schema.__name__,
        ),
        output_type=schema,
        tools=[],
        handoffs=[],
    )


# --- Agent 1: Domain Identifier ---
domain_identifier_agent = Agent(
    name="DomainIdentifierAgent",
    instructions=(
        "Your primary task: Analyze the provided text content and identify the single, most relevant high-level domain. "
        "Examples include: Finance, Technology, Healthcare, Arts, Science, Entertainment, Sports, "
        "Politics, Education, Environment, Business, Lifestyle, Travel, etc. "
        "Focus on the *primary* topic. The 'domain' field must contain a single concise label representing this dominant topic.\n"
        "If several potential domains appear in the text, select the one with the greatest overall coverage.\n"
        "Do **not** provide `confidence_score`, `relevance_score`, or `clarity_score`; these will be produced later.\n"
        "Output ONLY valid JSON matching the DomainSchema."
    ),
    model=DOMAIN_MODEL,
    output_type=DomainSchema,
    tools=[],
    handoffs=[],
)

# --- Agent 1b: Domain Result ---
domain_result_agent = create_result_agent(
    base_agent=domain_identifier_agent,
    schema=DomainResultSchema,
    item_description="domain label",
)

# --- Agent 2: Sub-Domain Identifier ---
sub_domain_identifier_agent = Agent(
    name="SubDomainIdentifierAgent",
    instructions=(
        "You are given text content and its primary domain. Your task is to identify specific sub-domains "
        "within the text related to the primary domain. "
        "Also provide a brief overall analysis summary.\n"
        "Output ONLY the result using the provided SubDomainIdentifierSchema."
    ),
    model=SUB_DOMAIN_MODEL,
    tools=[],
    handoffs=[],
    output_type=SubDomainIdentifierSchema,
)

# --- Agent 2b: Sub-Domain Result ---
sub_domain_result_agent = create_result_agent(
    base_agent=sub_domain_identifier_agent,
    schema=SubDomainSchema,
    item_description="sub-domain list",
)

# --- Agent 3: Topic Identifier (for one sub-domain) ---
topic_identifier_agent = Agent(
    name="TopicIdentifierAgent",
    instructions=(
        "You are provided with text, its primary domain, and ONE specific sub-domain. "
        "Your task: Analyze the *full text* and identify specific, relevant topics mentioned within the text that fall under the provided single sub-domain. "
        "Call the confidence_score, relevance_score, and clarity_score tools for each identified topic before producing the final output.\n"
        "Output the results ONLY using the provided SingleSubDomainTopicSchema. Every item in identified_topics MUST include the topic string plus 'confidence_score', 'relevance_score', and 'clarity_score'."
    ),
    model=TOPIC_MODEL,
    tools=[
        confidence_score_agent.as_tool(
            tool_name="confidence_score",
            tool_description="Evaluate confidence between 0.0 and 1.0",
        ),
        relevance_score_agent.as_tool(
            tool_name="relevance_score",
            tool_description="Judge relevance between 0.0 and 1.0",
        ),
        clarity_score_agent.as_tool(
            tool_name="clarity_score",
            tool_description="Assess clarity between 0.0 and 1.0",
        ),
    ],
    handoffs=[],
    output_type=SingleSubDomainTopicSchema,
)

# --- Agent 3b: Topic Result ---
topic_result_agent = create_result_agent(
    base_agent=topic_identifier_agent,
    schema=TopicSchema,
    item_description="topic analysis result",
)


# --- Base Agent for Type Identification (Agents 4a-4g) ---
# This base agent provides a template for identifying various concept types.
# It will be cloned and specialized for each specific type.

base_type_identifier_instructions_template = (
    "Your primary task: Analyze the provided text content to identify key {concept_description}. {specific_constraint} "
    "You will be given the full text AND context about its primary domain, identified sub-domains, and relevant topics found in previous analysis steps.\n"
    "Use this context to assess which {concept_type_singular}s are most relevant to the overall subject matter.\n"
    "For EACH identified {concept_type_singular}, provide:\n"
    "1. The classified {concept_type_singular}.\n"
    "Do **not** include scoring fields such as 'confidence_score', 'relevance_score', or 'clarity_score'.\n"
    "Provide an overall analysis summary if applicable.\n"
    "Output ONLY the result using the provided schema structure. Ensure the {list_field_name} field contains a list of items, each with '{item_field_name}'. Include the 'primary_domain' and 'analyzed_sub_domains' fields from the context in your output schema."
)

base_type_identifier_agent = Agent(
    name="BaseTypeIdentifierAgent",  # Generic name, will be overridden
    instructions=base_type_identifier_instructions_template,  # Will be formatted in clones
    # No default model or output_type, must be specified in clones
    tools=[],
    handoffs=[],
)


# --- Agent 4a: Entity Type Identifier ---
entity_type_identifier_agent = base_type_identifier_agent.clone(
    name="EntityTypeIdentifierAgent",
    instructions=base_type_identifier_instructions_template.format(
        concept_description="entity types (e.g., PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PRODUCT, TECHNOLOGY, SCIENTIFIC_CONCEPT, ECONOMIC_INDICATOR)",
        specific_constraint="Do NOT identify Event types - that is handled by another agent.",  # Retain original constraint
        concept_type_singular="entity type",
        list_field_name="identified_entities",
        item_field_name="entity_type",
    ),
    model=ENTITY_TYPE_MODEL,
    output_type=EntityTypeIdentifierSchema,
)

# --- Agent 4b: Ontology Type Identifier ---
ontology_type_identifier_agent = base_type_identifier_agent.clone(
    name="OntologyTypeIdentifierAgent",
    instructions=base_type_identifier_instructions_template.format(
        concept_description="relevant ontology types or concepts, potentially referencing standard ontologies (like Schema.org, FIBO, domain-specific ones) where applicable",
        specific_constraint="Focus on conceptual or taxonomic classifications, potentially referencing standard ontologies (like Schema.org, FIBO). Avoid simple entity labels.",  # Added constraint
        concept_type_singular="ontology type/concept",
        list_field_name="identified_ontology_types",
        item_field_name="ontology_type",
    ),
    model=ONTOLOGY_TYPE_MODEL,
    output_type=OntologyTypeIdentifierSchema,
)

# --- Agent 4c: Event Type Identifier ---
event_type_identifier_agent = base_type_identifier_agent.clone(
    name="EventTypeIdentifierAgent",
    instructions=base_type_identifier_instructions_template.format(
        concept_description="key EVENT types (e.g., Meeting, Acquisition, Conference, Product Launch, Election, Natural Disaster, Release, Protest, Accident, Celebration)",
        specific_constraint="Do NOT identify other entity types like Person, Organization, Location etc. - focus ONLY on events.",  # Retain original constraint
        concept_type_singular="event type",
        list_field_name="identified_events",
        item_field_name="event_type",
    ),
    model=EVENT_TYPE_MODEL,
    output_type=EventTypeIdentifierSchema,
)

# --- Agent 4d: Statement Type Identifier ---
statement_type_identifier_agent = base_type_identifier_agent.clone(
    name="StatementTypeIdentifierAgent",
    instructions=base_type_identifier_instructions_template.format(
        concept_description="key STATEMENT types (e.g., Fact, Claim, Opinion, Question, Instruction, Hypothesis, Prediction)",
        specific_constraint="Focus only on classifying the nature or type of the statement (e.g., Fact, Opinion, Claim, Hypothesis), not its specific content or truth value.",  # Added constraint
        concept_type_singular="statement type",
        list_field_name="identified_statements",
        item_field_name="statement_type",
    ),
    model=STATEMENT_TYPE_MODEL,
    output_type=StatementTypeIdentifierSchema,
)

# --- Agent 4e: Evidence Type Identifier ---
evidence_type_identifier_agent = base_type_identifier_agent.clone(
    name="EvidenceTypeIdentifierAgent",
    instructions=base_type_identifier_instructions_template.format(
        concept_description="key types of EVIDENCE presented (e.g., Testimony, Document Reference, Statistic, Anecdote, Expert Opinion, Observation, Example, Case Study, Logical Argument)",
        specific_constraint="Focus on the *form* or *category* of evidence used to support claims or statements (e.g., Statistic, Testimony, Document Reference).",  # Added constraint
        concept_type_singular="evidence type",
        list_field_name="identified_evidence",
        item_field_name="evidence_type",
    ),
    model=EVIDENCE_TYPE_MODEL,
    output_type=EvidenceTypeIdentifierSchema,
)

# --- Agent 4f: Measurement Type Identifier ---
measurement_type_identifier_agent = base_type_identifier_agent.clone(
    name="MeasurementTypeIdentifierAgent",
    instructions=base_type_identifier_instructions_template.format(
        concept_description="key types of MEASUREMENTS mentioned (e.g., Financial Metric, Physical Quantity, Performance Indicator, Survey Result, Count, Ratio, Percentage, Score)",
        specific_constraint="Focus on the *category* or *type* of measurement being used (e.g., Financial Metric, Physical Quantity, Ratio), not necessarily the specific values.",  # Added constraint
        concept_type_singular="measurement type",
        list_field_name="identified_measurements",
        item_field_name="measurement_type",
    ),
    model=MEASUREMENT_TYPE_MODEL,
    output_type=MeasurementTypeIdentifierSchema,
)

# --- Agent 4g: Modality Type Identifier ---
modality_type_identifier_agent = base_type_identifier_agent.clone(
    name="ModalityTypeIdentifierAgent",
    instructions=base_type_identifier_instructions_template.format(
        concept_description="the types of MODALITIES represented or referred to (e.g., Text, Image, Video, Audio, Table, Chart, Code Snippet, Mathematical Formula, Diagram)",
        specific_constraint="Identify the *format or medium* of information presented or referenced (e.g., Text, Image, Table, Code Snippet).",  # Added constraint
        concept_type_singular="modality type",
        list_field_name="identified_modalities",
        item_field_name="modality_type",
    ),
    model=MODALITY_TYPE_MODEL,
    output_type=ModalityTypeIdentifierSchema,
)


# --- Base Agent for Instance Extraction (Agents 5a-5g & 6b) ---
# Provides a reusable template for extracting specific instances of the previously
# identified types. Specific extractor agents clone this base and customize the
# placeholders for their schema fields.

base_instance_extractor_instructions_template = (
    "Extract specific {concept_description} from the provided text. "
    "Use the context of domain, sub-domains, topics and identified {type_list_name} to guide relevance. "
    "For each extracted instance provide the {instance_field} and {span_field}. "
    "Do **not** include scoring fields such as 'confidence_score', 'relevance_score', or 'clarity_score'. "
    "Output ONLY the result using the provided schema structure. "
    "Ensure the '{list_field}' field contains all extracted items and include the 'primary_domain' and 'analyzed_sub_domains' fields from the context."
)

base_instance_extractor_agent = Agent(
    name="BaseInstanceExtractorAgent",  # Generic name, overridden in clones
    instructions=base_instance_extractor_instructions_template,  # Formatted in clones
    tools=[],
    handoffs=[],
)


# --- Agent 5: Entity Instance Extractor ---
# Clone of base_instance_extractor_agent specialized for entity mentions.
entity_instance_extractor_agent = base_instance_extractor_agent.clone(
    name="EntityInstanceExtractorAgent",
    instructions=base_instance_extractor_instructions_template.format(
        concept_description="entity mentions",
        type_list_name="entity types",
        instance_field="entity type",
        span_field="exact text span and character offsets",
        list_field="identified_instances",
    ),
    model=ENTITY_INSTANCE_MODEL,
    output_type=EntityInstanceSchema,
)


# --- Agent 5b: Ontology Instance Extractor ---
# Clone of base_instance_extractor_agent specialized for ontology concepts.
ontology_instance_extractor_agent = base_instance_extractor_agent.clone(
    name="OntologyInstanceExtractorAgent",
    instructions=base_instance_extractor_instructions_template.format(
        concept_description="ontology concept mentions",
        type_list_name="ontology types",
        instance_field="ontology type",
        span_field="exact text span and character offsets",
        list_field="identified_instances",
    ),
    model=ONTOLOGY_INSTANCE_MODEL,
    output_type=OntologyInstanceSchema,
)


# --- Agent 5c: Event Instance Extractor ---
# Clone of base_instance_extractor_agent specialized for event mentions.
event_instance_extractor_agent = base_instance_extractor_agent.clone(
    name="EventInstanceExtractorAgent",
    instructions=base_instance_extractor_instructions_template.format(
        concept_description="event mentions",
        type_list_name="event types",
        instance_field="event type",
        span_field="exact text span and character offsets",
        list_field="identified_instances",
    ),
    model=EVENT_INSTANCE_MODEL,
    output_type=EventInstanceSchema,
)


# --- Agent 5d: Statement Instance Extractor ---
# Clone of base_instance_extractor_agent specialized for statement snippets.
statement_instance_extractor_agent = base_instance_extractor_agent.clone(
    name="StatementInstanceExtractorAgent",
    instructions=base_instance_extractor_instructions_template.format(
        concept_description="statement snippets",
        type_list_name="statement types",
        instance_field="statement type",
        span_field="exact text span and character offsets",
        list_field="identified_instances",
    ),
    model=STATEMENT_INSTANCE_MODEL,
    output_type=StatementInstanceSchema,
)


# --- Agent 5e: Evidence Instance Extractor ---
# Clone of base_instance_extractor_agent specialized for evidence mentions.
evidence_instance_extractor_agent = base_instance_extractor_agent.clone(
    name="EvidenceInstanceExtractorAgent",
    instructions=base_instance_extractor_instructions_template.format(
        concept_description="evidence mentions",
        type_list_name="evidence types",
        instance_field="evidence type",
        span_field="exact text span and character offsets",
        list_field="identified_instances",
    ),
    model=EVIDENCE_INSTANCE_MODEL,
    output_type=EvidenceInstanceSchema,
)


# --- Agent 5f: Measurement Instance Extractor ---
# Clone of base_instance_extractor_agent specialized for measurement mentions.
measurement_instance_extractor_agent = base_instance_extractor_agent.clone(
    name="MeasurementInstanceExtractorAgent",
    instructions=base_instance_extractor_instructions_template.format(
        concept_description="measurement mentions",
        type_list_name="measurement types",
        instance_field="measurement type",
        span_field="exact text span and character offsets",
        list_field="identified_instances",
    ),
    model=MEASUREMENT_INSTANCE_MODEL,
    output_type=MeasurementInstanceSchema,
)


# --- Agent 5g: Modality Instance Extractor ---
# Clone of base_instance_extractor_agent specialized for modality references.
modality_instance_extractor_agent = base_instance_extractor_agent.clone(
    name="ModalityInstanceExtractorAgent",
    instructions=base_instance_extractor_instructions_template.format(
        concept_description="modality references",
        type_list_name="modality types",
        instance_field="modality type",
        span_field="exact text span and character offsets",
        list_field="identified_instances",
    ),
    model=MODALITY_INSTANCE_MODEL,
    output_type=ModalityInstanceSchema,
)


# --- Agent 6: Relationship Identifier (for one entity type) ---
relationship_type_identifier_agent = Agent(
    name="RelationshipTypeIdentifierAgent",
    instructions=(
        "Your task: Analyze the provided text and context (domain, sub-domains, topics, and entity types) to identify relationships between entities. "
        "You will be given the *full text* and context, PLUS **one specific entity type** (identified in a previous step) to focus on (e.g., 'ORGANIZATION').\n"
        "Identify explicit or strongly implied relationships mentioned in the text where the **focus entity type** is involved as one of the participants.\n"
        "Examples of relationships: WORKS_FOR, LOCATED_IN, ACQUIRED, PARTNERED_WITH, COMPETES_WITH, FOUNDED_BY, MANUFACTURES, USES_TECHNOLOGY, etc.\n"
        "For EACH identified relationship involving the focus entity type:\n"
        "1. State the unique type of relationship found.\n"
        "2. Call the confidence_score, relevance_score, and clarity_score tools to score the relationship before producing the final output.\n"
        "Output ONLY the result using the provided SingleEntityTypeRelationshipSchema. Ensure the 'entity_type_focus' field matches the entity type you were asked to focus on. Every RelationshipDetail in the 'identified_relationships' list MUST include 'confidence_score', 'relevance_score', and 'clarity_score'. Do not add commentary outside the schema."
    ),
    model=RELATIONSHIP_MODEL,
    output_type=SingleEntityTypeRelationshipSchema,
    tools=[
        confidence_score_agent.as_tool(
            tool_name="confidence_score",
            tool_description="Evaluate confidence between 0.0 and 1.0",
        ),
        relevance_score_agent.as_tool(
            tool_name="relevance_score",
            tool_description="Judge relevance between 0.0 and 1.0",
        ),
        clarity_score_agent.as_tool(
            tool_name="clarity_score",
            tool_description="Assess clarity between 0.0 and 1.0",
        ),
    ],
    handoffs=[],
)

# --- Agent 6b: Relationship Instance Extractor ---
# Clone of base_instance_extractor_agent specialized for relationship instances.
relationship_extractor_agent = base_instance_extractor_agent.clone(
    name="RelationshipInstanceExtractorAgent",
    instructions=base_instance_extractor_instructions_template.format(
        concept_description="subject-object relationships",
        type_list_name="relationship types and extracted entity instances",
        instance_field="subject, relationship type, object, and relevance score",
        span_field="optional snippet",
        list_field="identified_instances",
    ),
    model=RELATIONSHIP_INSTANCE_MODEL,
    output_type=RelationshipInstanceSchema,
)

# You can optionally create a list or dict to easily access all agents
all_agents = {
    "domain_identifier": domain_identifier_agent,
    "domain_result": domain_result_agent,
    "sub_domain_identifier": sub_domain_identifier_agent,
    "sub_domain_result": sub_domain_result_agent,
    "topic_identifier": topic_identifier_agent,
    "topic_result": topic_result_agent,
    "entity_type_identifier": entity_type_identifier_agent,
    "ontology_type_identifier": ontology_type_identifier_agent,
    "event_type_identifier": event_type_identifier_agent,
    "statement_type_identifier": statement_type_identifier_agent,
    "evidence_type_identifier": evidence_type_identifier_agent,
    "measurement_type_identifier": measurement_type_identifier_agent,
    "modality_type_identifier": modality_type_identifier_agent,
    "entity_instance_extractor": entity_instance_extractor_agent,
    "ontology_instance_extractor": ontology_instance_extractor_agent,
    "event_instance_extractor": event_instance_extractor_agent,
    "statement_instance_extractor": statement_instance_extractor_agent,
    "evidence_instance_extractor": evidence_instance_extractor_agent,
    "measurement_instance_extractor": measurement_instance_extractor_agent,
    "modality_instance_extractor": modality_instance_extractor_agent,
    "confidence_score": confidence_score_agent,
    "relevance_score": relevance_score_agent,
    "clarity_score": clarity_score_agent,
    "relationship_identifier": relationship_type_identifier_agent,
    "relationship_instance_extractor": relationship_extractor_agent,
    # Note: Base agent is not typically included here unless used directly
}

if "__all__" in globals():
    __all_list = cast(List[str], globals()["__all__"])
    __all_list.append("domain_result_agent")
    __all_list.append("sub_domain_result_agent")
    __all_list.append("topic_result_agent")
