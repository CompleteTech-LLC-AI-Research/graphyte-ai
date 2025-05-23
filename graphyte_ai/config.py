import os
import sys
import logging
from pathlib import Path
import importlib.util

# --- Third-Party Imports ---
# Environment Variable Loading (using python-dotenv)
try:
    from dotenv import load_dotenv

    # Try loading from the project root
    dotenv_path_project_root = Path(__file__).resolve().parent / ".env"
    if dotenv_path_project_root.exists():
        load_dotenv(dotenv_path=dotenv_path_project_root)
        print(f"Loaded environment variables from: {dotenv_path_project_root}")
    else:
        # Fallback to current working directory (less common for packages)
        dotenv_path_cwd = Path(".env")
        if dotenv_path_cwd.exists():
            load_dotenv(dotenv_path=dotenv_path_cwd)
            print(
                f"Loaded environment variables from current directory: {dotenv_path_cwd}"
            )
        else:
            print(
                f"Warning: .env file not found at {dotenv_path_project_root} or current directory. Environment variables should be set manually."
            )
except ImportError:
    print(
        "Warning: 'python-dotenv' not found. Install with 'pip install python-dotenv'. Environment variables should be set manually."
    )

# --- SDK Imports ---
# NOTE: Using the external ``agents`` SDK
try:
    from agents import set_default_openai_key  # type: ignore[attr-defined]
except ImportError:
    print(
        "Error: 'agents' SDK library not found or incomplete. Please ensure it is installed and accessible."
    )
    sys.exit(1)

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR_BASE = PROJECT_ROOT / "outputs"
DOMAIN_OUTPUT_DIR = OUTPUTS_DIR_BASE / "01_domain_identifier"
SUB_DOMAIN_OUTPUT_DIR = OUTPUTS_DIR_BASE / "02_sub_domain_identifier"
TOPIC_OUTPUT_DIR = OUTPUTS_DIR_BASE / "03_topic_identifier"
ENTITY_TYPE_OUTPUT_DIR = OUTPUTS_DIR_BASE / "04a_entity_type_identifier"
ONTOLOGY_TYPE_OUTPUT_DIR = OUTPUTS_DIR_BASE / "04b_ontology_type_identifier"
EVENT_TYPE_OUTPUT_DIR = OUTPUTS_DIR_BASE / "04c_event_type_identifier"
STATEMENT_TYPE_OUTPUT_DIR = OUTPUTS_DIR_BASE / "04d_statement_type_identifier"
EVIDENCE_TYPE_OUTPUT_DIR = OUTPUTS_DIR_BASE / "04e_evidence_type_identifier"
MEASUREMENT_TYPE_OUTPUT_DIR = OUTPUTS_DIR_BASE / "04f_measurement_type_identifier"
MODALITY_TYPE_OUTPUT_DIR = (
    OUTPUTS_DIR_BASE / "04g_modality_type_identifier"
)  # Added directory for new agent (4g)
ENTITY_INSTANCE_OUTPUT_DIR = OUTPUTS_DIR_BASE / "05a_entity_instance_extractor"
ONTOLOGY_INSTANCE_OUTPUT_DIR = OUTPUTS_DIR_BASE / "05b_ontology_instance_extractor"
EVENT_INSTANCE_OUTPUT_DIR = OUTPUTS_DIR_BASE / "05c_event_instance_extractor"
STATEMENT_INSTANCE_OUTPUT_DIR = OUTPUTS_DIR_BASE / "05d_statement_instance_extractor"
EVIDENCE_INSTANCE_OUTPUT_DIR = OUTPUTS_DIR_BASE / "05e_evidence_instance_extractor"
MEASUREMENT_INSTANCE_OUTPUT_DIR = (
    OUTPUTS_DIR_BASE / "05f_measurement_instance_extractor"
)
MODALITY_INSTANCE_OUTPUT_DIR = OUTPUTS_DIR_BASE / "05g_modality_instance_extractor"
AGGREGATED_INSTANCE_OUTPUT_DIR = OUTPUTS_DIR_BASE / "06c_aggregated_instances"
RELATIONSHIP_OUTPUT_DIR = OUTPUTS_DIR_BASE / "06_relationship_identifier"
RELATIONSHIP_INSTANCE_OUTPUT_DIR = (
    OUTPUTS_DIR_BASE / "06b_relationship_instance_extractor"
)
DOMAIN_OUTPUT_FILENAME = "domain_identifier_output.json"
SUB_DOMAIN_OUTPUT_FILENAME = "sub_domain_identifier_output.json"
TOPIC_OUTPUT_FILENAME = "topic_identifier_output.json"
ENTITY_TYPE_OUTPUT_FILENAME = "entity_type_identifier_output.json"
ONTOLOGY_TYPE_OUTPUT_FILENAME = "ontology_type_identifier_output.json"
EVENT_TYPE_OUTPUT_FILENAME = "event_type_identifier_output.json"
STATEMENT_TYPE_OUTPUT_FILENAME = "statement_type_identifier_output.json"
EVIDENCE_TYPE_OUTPUT_FILENAME = "evidence_type_identifier_output.json"
MEASUREMENT_TYPE_OUTPUT_FILENAME = "measurement_type_identifier_output.json"
MODALITY_TYPE_OUTPUT_FILENAME = (
    "modality_type_identifier_output.json"  # Added filename for new agent (4g)
)
ENTITY_INSTANCE_OUTPUT_FILENAME = "entity_instance_extractor_output.json"
ONTOLOGY_INSTANCE_OUTPUT_FILENAME = "ontology_instance_extractor_output.json"
EVENT_INSTANCE_OUTPUT_FILENAME = "event_instance_extractor_output.json"
STATEMENT_INSTANCE_OUTPUT_FILENAME = "statement_instance_extractor_output.json"
EVIDENCE_INSTANCE_OUTPUT_FILENAME = "evidence_instance_extractor_output.json"
MEASUREMENT_INSTANCE_OUTPUT_FILENAME = "measurement_instance_extractor_output.json"
MODALITY_INSTANCE_OUTPUT_FILENAME = "modality_instance_extractor_output.json"
AGGREGATED_INSTANCE_OUTPUT_FILENAME = "aggregated_instance_output.json"
RELATIONSHIP_OUTPUT_FILENAME = "relationship_identifier_output.json"
RELATIONSHIP_INSTANCE_OUTPUT_FILENAME = "relationship_instance_extractor_output.json"
VISUALIZATION_OUTPUT_DIR = OUTPUTS_DIR_BASE / "00_visualization"
VISUALIZATION_FILENAME = "agent_workflow_graph.gv"

# Common binary file extensions to skip during directory processing
BINARY_FILE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    # '.pdf', # Keep PDF, handled separately by read_input_from_file in utils
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".zip",
    ".gz",
    ".tar",
    ".bz2",
    ".rar",
    ".7z",
    ".exe",
    ".dll",
    ".so",
    ".o",
    ".a",
    ".dylib",
    ".class",
    ".pyc",
    ".jar",
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".aac",
    ".mp4",
    ".avi",
    ".mov",
    ".wmv",
    ".mkv",
    ".sqlite",
    ".db",
    ".mdb",
}

# Default model to use if environment variables are not set
DEFAULT_MODEL = "gpt-4o-mini"
# Threshold for warning about large input content size
MAX_INPUT_CONTENT_LENGTH = 1_000_000  # Warn if input exceeds 1 million characters

# Check optional dependencies availability (useful for utils module)
PYMUPDF_AVAILABLE = importlib.util.find_spec("fitz") is not None
TENACITY_AVAILABLE = importlib.util.find_spec("tenacity") is not None

# --- Configuration Loading ---
# Load model names from environment variables, falling back to the default
DOMAIN_MODEL = os.getenv("DOMAIN_IDENTIFIER_MODEL", DEFAULT_MODEL)
SUB_DOMAIN_MODEL = os.getenv("SUB_DOMAIN_IDENTIFIER_MODEL", DEFAULT_MODEL)
TOPIC_MODEL = os.getenv("TOPIC_IDENTIFIER_MODEL", DEFAULT_MODEL)
ENTITY_TYPE_MODEL = os.getenv("ENTITY_TYPE_IDENTIFIER_MODEL", DEFAULT_MODEL)
ONTOLOGY_TYPE_MODEL = os.getenv("ONTOLOGY_TYPE_IDENTIFIER_MODEL", DEFAULT_MODEL)
EVENT_TYPE_MODEL = os.getenv("EVENT_TYPE_IDENTIFIER_MODEL", DEFAULT_MODEL)
STATEMENT_TYPE_MODEL = os.getenv("STATEMENT_TYPE_IDENTIFIER_MODEL", DEFAULT_MODEL)
EVIDENCE_TYPE_MODEL = os.getenv("EVIDENCE_TYPE_IDENTIFIER_MODEL", DEFAULT_MODEL)
MEASUREMENT_TYPE_MODEL = os.getenv("MEASUREMENT_TYPE_IDENTIFIER_MODEL", DEFAULT_MODEL)
MODALITY_TYPE_MODEL = os.getenv(
    "MODALITY_TYPE_IDENTIFIER_MODEL", DEFAULT_MODEL
)  # Added model for new agent (4g)
ENTITY_INSTANCE_MODEL = os.getenv("ENTITY_INSTANCE_EXTRACTOR_MODEL", DEFAULT_MODEL)
ONTOLOGY_INSTANCE_MODEL = os.getenv("ONTOLOGY_INSTANCE_EXTRACTOR_MODEL", DEFAULT_MODEL)
EVENT_INSTANCE_MODEL = os.getenv("EVENT_INSTANCE_EXTRACTOR_MODEL", DEFAULT_MODEL)
STATEMENT_INSTANCE_MODEL = os.getenv(
    "STATEMENT_INSTANCE_EXTRACTOR_MODEL", DEFAULT_MODEL
)
EVIDENCE_INSTANCE_MODEL = os.getenv("EVIDENCE_INSTANCE_EXTRACTOR_MODEL", DEFAULT_MODEL)
MEASUREMENT_INSTANCE_MODEL = os.getenv(
    "MEASUREMENT_INSTANCE_EXTRACTOR_MODEL", DEFAULT_MODEL
)
MODALITY_INSTANCE_MODEL = os.getenv("MODALITY_INSTANCE_EXTRACTOR_MODEL", DEFAULT_MODEL)
RELATIONSHIP_MODEL = os.getenv("RELATIONSHIP_IDENTIFIER_MODEL", DEFAULT_MODEL)
RELATIONSHIP_INSTANCE_MODEL = os.getenv(
    "RELATIONSHIP_INSTANCE_EXTRACTOR_MODEL", DEFAULT_MODEL
)
# Load OpenAI API Key from environment variable
API_KEY = os.getenv("OPENAI_API_KEY")
# Load optional base URL for tracing platform
AGENT_TRACE_BASE_URL = os.getenv(
    "AGENT_TRACE_BASE_URL", "https://platform.openai.com/traces"
)  # Example, adjust if needed

# --- API Key Setup ---
# CRITICAL: Check for API Key early and exit if missing
if not API_KEY:
    try:
        # Attempt to log the critical error before exiting (basic setup here)
        logger_early = logging.getLogger(__name__)
        logger_early.addHandler(logging.StreamHandler())
        logger_early.setLevel(logging.CRITICAL)
        logger_early.critical("CRITICAL: OPENAI_API_KEY environment variable not set.")
    except Exception:
        # Fallback to simple print if logging fails
        print(
            "CRITICAL ERROR: OPENAI_API_KEY environment variable not set.",
            file=sys.stderr,
        )
    sys.exit(1)

# Set API key for the ``agents`` SDK (or equivalent SDK function)
try:
    set_default_openai_key(API_KEY)
    print("OpenAI API Key set for agents SDK.")
except NameError:
    print(
        "Error: Could not set OpenAI key via agents SDK (likely import failure).",
        file=sys.stderr,
    )
    sys.exit(1)
except Exception as e:
    print(f"Error setting OpenAI key via agents SDK: {e}", file=sys.stderr)
    sys.exit(1)

# --- Initial Logger Check ---
# Setup basic logger configuration to catch early issues if logging setup in utils fails
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)  # Get logger for this config module

# Log status of optional dependencies early
if not PYMUPDF_AVAILABLE:
    logger.warning(
        "PyMuPDF (fitz) library not found. PDF file processing will be skipped."
    )
if not TENACITY_AVAILABLE:
    logger.warning(
        "Tenacity library not found. Retry logic for agent runs will be disabled."
    )
