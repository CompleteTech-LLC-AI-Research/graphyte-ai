# File: /Users/completetech/Desktop/python-agent-sdk/src/agentic_team_workflow/config.py
import os
import sys
import logging
from pathlib import Path

# --- Third-Party Imports ---
# Environment Variable Loading (using python-dotenv)
try:
    from dotenv import load_dotenv
    # Try loading from parent's parent first (common project structure)
    # Assuming this file is in agentic_team_workflow, parent is project_root
    dotenv_path_project_root = Path(__file__).resolve().parent.parent / '.env'
    if dotenv_path_project_root.exists():
        load_dotenv(dotenv_path=dotenv_path_project_root)
        print(f"Loaded environment variables from: {dotenv_path_project_root}")
    else:
        # Fallback to current working directory (less common for packages)
        dotenv_path_cwd = Path('.env')
        if dotenv_path_cwd.exists():
            load_dotenv(dotenv_path=dotenv_path_cwd)
            print(f"Loaded environment variables from current directory: {dotenv_path_cwd}")
        else:
            print(f"Warning: .env file not found at {dotenv_path_project_root} or current directory. Environment variables should be set manually.")
except ImportError:
    print("Warning: 'python-dotenv' not found. Install with 'pip install python-dotenv'. Environment variables should be set manually.")

# --- SDK Imports ---
# NOTE: Using 'agentic_team' as the alias for the SDK import
try:
    from agentic_team import set_default_openai_key
except ImportError:
    print("Error: 'agentic_team' SDK library not found or incomplete. Please ensure it is installed and accessible.")
    sys.exit(1)

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
MODALITY_TYPE_OUTPUT_DIR = OUTPUTS_DIR_BASE / "04g_modality_type_identifier" # Added directory for new agent (4g)
ENTITY_INSTANCE_OUTPUT_DIR = OUTPUTS_DIR_BASE / "05_entity_instance_extractor"
RELATIONSHIP_OUTPUT_DIR = OUTPUTS_DIR_BASE / "06_relationship_identifier"
DOMAIN_OUTPUT_FILENAME = "domain_identifier_output.json"
SUB_DOMAIN_OUTPUT_FILENAME = "sub_domain_identifier_output.json"
TOPIC_OUTPUT_FILENAME = "topic_identifier_output.json"
ENTITY_TYPE_OUTPUT_FILENAME = "entity_type_identifier_output.json"
ONTOLOGY_TYPE_OUTPUT_FILENAME = "ontology_type_identifier_output.json"
EVENT_TYPE_OUTPUT_FILENAME = "event_type_identifier_output.json"
STATEMENT_TYPE_OUTPUT_FILENAME = "statement_type_identifier_output.json"
EVIDENCE_TYPE_OUTPUT_FILENAME = "evidence_type_identifier_output.json"
MEASUREMENT_TYPE_OUTPUT_FILENAME = "measurement_type_identifier_output.json"
MODALITY_TYPE_OUTPUT_FILENAME = "modality_type_identifier_output.json" # Added filename for new agent (4g)
ENTITY_INSTANCE_OUTPUT_FILENAME = "entity_instance_extractor_output.json"
RELATIONSHIP_OUTPUT_FILENAME = "relationship_identifier_output.json"
VISUALIZATION_OUTPUT_DIR = OUTPUTS_DIR_BASE / "00_visualization"
VISUALIZATION_FILENAME = "agent_workflow_graph.gv"

# Common binary file extensions to skip during directory processing
BINARY_FILE_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff',
    # '.pdf', # Keep PDF, handled separately by read_input_from_file in utils
    '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.zip', '.gz', '.tar', '.bz2', '.rar', '.7z',
    '.exe', '.dll', '.so', '.o', '.a', '.dylib',
    '.class', '.pyc', '.jar',
    '.mp3', '.wav', '.ogg', '.flac', '.aac',
    '.mp4', '.avi', '.mov', '.wmv', '.mkv',
    '.sqlite', '.db', '.mdb'
}

# Default model to use if environment variables are not set
DEFAULT_MODEL = "gpt-4o-mini"
# Threshold for warning about large input content size
MAX_INPUT_CONTENT_LENGTH = 1_000_000 # Warn if input exceeds 1 million characters

# Check optional dependencies availability (useful for utils module)
try:
    import fitz # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

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
MODALITY_TYPE_MODEL = os.getenv("MODALITY_TYPE_IDENTIFIER_MODEL", DEFAULT_MODEL) # Added model for new agent (4g)
ENTITY_INSTANCE_MODEL = os.getenv("ENTITY_INSTANCE_EXTRACTOR_MODEL", DEFAULT_MODEL)
RELATIONSHIP_MODEL = os.getenv("RELATIONSHIP_IDENTIFIER_MODEL", DEFAULT_MODEL)
# Load OpenAI API Key from environment variable
API_KEY = os.getenv("OPENAI_API_KEY")
# Load optional base URL for tracing platform
AGENT_TRACE_BASE_URL = os.getenv("AGENT_TRACE_BASE_URL", "https://platform.openai.com/traces") # Example, adjust if needed

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
        print("CRITICAL ERROR: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
    sys.exit(1)

# Set API key for the 'agentic_team' SDK (or equivalent SDK function)
try:
    set_default_openai_key(API_KEY)
    print("OpenAI API Key set for agentic_team SDK.")
except NameError:
    print("Error: Could not set OpenAI key via agentic_team SDK (likely import failure).", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error setting OpenAI key via agentic_team SDK: {e}", file=sys.stderr)
    sys.exit(1)

# --- Initial Logger Check ---
# Setup basic logger configuration to catch early issues if logging setup in utils fails
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__) # Get logger for this config module

# Log status of optional dependencies early
if not PYMUPDF_AVAILABLE:
    logger.warning("PyMuPDF (fitz) library not found. PDF file processing will be skipped.")
if not TENACITY_AVAILABLE:
    logger.warning("Tenacity library not found. Retry logic for agent runs will be disabled.")