import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from pydantic import ValidationError

# Conditional Imports for Optional Features
try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
    )

    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

# NOTE: Using the external ``agents`` SDK
try:
    from agents import (  # type: ignore[attr-defined]
        Agent,
        Runner,
        RunConfig,
        TResponseInputItem,
        RunResult,
        AgentsException,  # Import base SDK exception for retry
    )
except ImportError:
    print("Error: 'agents' SDK library not found or incomplete. Cannot define utils.")
    raise

# Import config constants needed in utils
from .config import (
    LOGS_DIR,
    PROJECT_ROOT,
    BINARY_FILE_EXTENSIONS,
)
from .workflow_agents import (
    confidence_score_agent,
    relevance_score_agent,
    clarity_score_agent,
)
from .schemas import (
    ConfidenceScoreSchema,
    RelevanceScoreSchema,
    ClarityScoreSchema,
    SubDomainSchema,
    TopicSchema,
    TopicDetail,
)

# Get logger for utils module
logger = logging.getLogger(__name__)


# --- Logging Setup ---
def setup_logging():
    """Configures logging for the application."""
    # Ensure the logs directory exists
    LOGS_DIR.mkdir(exist_ok=True)
    # Define the log file path
    log_file = LOGS_DIR / "workflow.log"  # Changed filename to be more generic
    # Remove existing handlers to avoid duplication if called multiple times
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,  # Set the minimum logging level (e.g., INFO, DEBUG)
        format="%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d]: %(message)s",
        handlers=[
            logging.FileHandler(
                log_file, mode="a", encoding="utf-8"
            ),  # Append to log file
            logging.StreamHandler(),  # Output to console (stderr by default)
        ],
        # Force=True might be useful in some complex scenarios, but avoid if possible
        # force=True
    )
    logger.info(f"Logging configured. Log file: {log_file}")


# --- Input Handling Functions ---


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for input source (file or directory) or visualization."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyzes text input to identify domain, sub-domains, topics, entities, and relationships. "
            "Can also generate a visualization of the agent structure."
        )
    )
    input_group = (
        parser.add_mutually_exclusive_group()
    )  # Group file/dir/stdin implicitly
    input_group.add_argument(
        "--file", type=str, help="Path to a single text file used as input."
    )
    input_group.add_argument(
        "--dir",
        type=str,
        help="Path to a directory containing text files used as input.",
    )
    # Stdin is the default if --file or --dir is not provided

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate a visualization of the agent workflow structure and exit.",
    )
    return parser.parse_args()


def read_input_from_file(file_path: Path) -> str:
    """Reads content from a single file. Attempts PDF text extraction if applicable."""
    if not file_path.is_file():
        logger.error(f"Input path is not a valid file: {file_path}")
        raise FileNotFoundError(f"Input path is not a valid file: {file_path}")

    logger.info(f"Reading input from file: {file_path}")

    if file_path.suffix.lower() == ".pdf":
        if not PYMUPDF_AVAILABLE:
            logger.error(
                f"Cannot read PDF '{file_path.name}'. 'PyMuPDF' library not found. Skipping."
            )
            raise ImportError(
                f"PyMuPDF not installed, cannot process PDF: {file_path.name}"
            )
        try:
            logger.info(f"Attempting to extract text from PDF: {file_path.name}")
            doc = fitz.open(file_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text("text")
            doc.close()
            if text and text.strip():
                logger.info(
                    f"Successfully extracted {len(text)} characters from PDF: {file_path.name}"
                )
                return text
            else:
                logger.warning(
                    f"Extracted empty or whitespace-only text from PDF: {file_path.name}. It might be image-based or corrupted."
                )
                return ""
        except Exception as e:
            logger.exception(f"Error reading PDF file {file_path}: {e}")
            raise IOError(f"Error reading PDF file {file_path}: {e}") from e
    else:
        try:
            encodings_to_try = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
            content = None
            for enc in encodings_to_try:
                try:
                    content = file_path.read_text(encoding=enc)
                    logger.debug(
                        f"Successfully read {file_path.name} with encoding {enc}"
                    )
                    return content
                except UnicodeDecodeError:
                    logger.debug(f"Failed to decode {file_path.name} with {enc}")
                    continue
            logger.warning(
                f"Could not decode {file_path.name} with standard encodings, trying with errors='replace'."
            )
            return file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.exception(f"Error reading non-PDF file {file_path}: {e}")
            raise IOError(f"Error reading file {file_path}: {e}") from e


def read_input_from_directory(dir_path: Path) -> str:
    """Reads and combines content from readable files (including PDFs) in a directory."""
    if not dir_path.is_dir():
        logger.error(f"Input path is not a valid directory: {dir_path}")
        raise NotADirectoryError(f"Input path is not a valid directory: {dir_path}")

    logger.info(f"Reading input from directory: {dir_path}")
    combined_content = []
    file_count = 0
    processed_files = 0
    skipped_binary = 0
    read_errors = 0

    try:
        files_to_process = sorted(list(dir_path.iterdir()))
    except OSError as e:
        logger.exception(f"Error listing files in directory {dir_path}: {e}")
        raise IOError(f"Error listing files in directory {dir_path}: {e}") from e

    for item_path in files_to_process:
        if item_path.is_file():
            processed_files += 1
            if (
                item_path.suffix.lower() in BINARY_FILE_EXTENSIONS
                and item_path.suffix.lower() != ".pdf"
            ):
                logger.debug(
                    f"Skipping potentially binary file (non-PDF): {item_path.name}"
                )
                skipped_binary += 1
                continue
            try:
                logger.debug(f"Processing file: {item_path}")
                text = read_input_from_file(item_path)
                if text and text.strip():
                    combined_content.append(
                        f"\n\n--- Content from: {item_path.name} ---\n\n{text}"
                    )
                    file_count += 1
                elif text is not None:
                    logger.debug(
                        f"File {item_path.name} resulted in empty or whitespace-only content."
                    )
                # Else: read_input_from_file would raise an error if None/failure

            except (IOError, ImportError, Exception) as e:
                read_errors += 1
                logger.warning(
                    f"Could not read or process file {item_path.name}: {type(e).__name__}: {e}"
                )
                if isinstance(e, ImportError):
                    logger.error(
                        f"ImportError likely due to missing dependency for {item_path.name}"
                    )
        elif item_path.is_dir():
            logger.info(f"Skipping subdirectory: {item_path.name}")
        else:
            logger.warning(f"Skipping non-file/non-directory item: {item_path.name}")

    if file_count == 0:
        logger.warning(
            f"No readable text files (or extractable PDFs) found or processed in directory: {dir_path} (out of {processed_files} items checked; {skipped_binary} skipped as binary; {read_errors} read errors)."
        )

    logger.info(f"Read content from {file_count} files in directory {dir_path}.")
    return "\n".join(combined_content)


def prompt_user_for_input() -> str:
    """Prompts the user to enter text via standard input (stdin)."""
    print("\nNo --file or --dir specified. Please enter your input text below.")
    print("End input with Ctrl+D (Unix/macOS) or Ctrl+Z then Enter (Windows).")
    print("-" * 20)
    try:
        lines = sys.stdin.readlines()
        if not lines:
            logger.warning("Received empty input from stdin.")
            return ""
        logger.info(f"Reading {len(lines)} lines of input from stdin.")
        return "".join(lines)
    except KeyboardInterrupt:
        print("\nInput cancelled by user.")
        logger.warning("Stdin input cancelled by user.")
        return ""


# --- Helper Function to Save JSON Output ---
def direct_save_json_output(
    output_dir: Path, filename: str, content: Dict[str, Any], trace_id: Optional[str]
) -> str:
    """Saves the provided dictionary content as a JSON file in the designated output directory."""
    safe_filename = Path(filename).name
    if not safe_filename:
        default_filename = f"output_{trace_id or 'unknown_trace'}.json"
        logger.warning(
            f"Original filename '{filename}' was invalid or empty, using default '{default_filename}'"
        )
        safe_filename = default_filename

    if not safe_filename.lower().endswith(".json"):
        safe_filename += ".json"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {output_dir}")
    except OSError as e:
        logger.exception(
            f"Failed to create output directory {output_dir}: {e}",
            extra={"trace_id": trace_id or "N/A"},
        )
        return f"Error creating output directory '{output_dir}' before saving {safe_filename}: {e}"

    output_path = output_dir / safe_filename
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)

        try:
            relative_output_dir = output_dir.relative_to(PROJECT_ROOT)
        except ValueError:
            relative_output_dir = output_dir  # Fallback to absolute path

        logger.info(
            f"Successfully saved data to file '{output_path.name}' in directory '{relative_output_dir}'."
        )
        return f"Success: Saved data to file '{output_path.name}' in directory '{relative_output_dir}'."

    except OSError as e:
        logger.exception(
            f"OS error saving file {output_path}: {e}",
            extra={"trace_id": trace_id or "N/A"},
        )
        return f"Error saving data to {safe_filename} due to OS error: {e}"
    except TypeError as e:
        logger.exception(
            f"Type error preparing data for JSON serialization to {output_path}: {e}",
            extra={"trace_id": trace_id or "N/A"},
        )
        return f"Error saving data to {safe_filename} due to data type issue: {e}"
    except Exception as e:
        logger.exception(
            f"Unexpected error saving file {output_path}: {e}",
            extra={"trace_id": trace_id or "N/A"},
        )
        return f"Error saving data to {safe_filename}: {e}"


# --- Retry Logic Setup ---
# Define a retry decorator if the 'tenacity' library is available
if TENACITY_AVAILABLE:
    logger.info("Tenacity library found. Enabling retry logic for agent runs.")
    retry_decorator = retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((AgentsException, asyncio.TimeoutError)),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying agent run for {retry_state.args[0].name if retry_state.args else 'unknown agent'} after error: {retry_state.outcome.exception()}. Attempt {retry_state.attempt_number+1}..."
        ),
    )

    @retry_decorator
    async def run_agent_with_retry(
        agent: Agent,
        input_data: Union[str, List[TResponseInputItem]],
        config: Optional[RunConfig] = None,
    ) -> RunResult:
        """Wrapper function to run an agent with configured retry logic."""
        logger.debug(f"Attempting to run agent '{agent.name}'...")
        result = await Runner.run(
            starting_agent=agent, input=input_data, run_config=config
        )
        logger.debug(f"Agent '{agent.name}' run successful.")
        return result

else:
    logger.warning(
        "Tenacity library not found. Agent runs will not have automatic retries."
    )

    async def run_agent_with_retry(
        agent: Agent,
        input_data: Union[str, List[TResponseInputItem]],
        config: Optional[RunConfig] = None,
    ) -> RunResult:
        """Placeholder function when tenacity is not available. Runs the agent once."""
        # No retry logic here, just call the runner directly.
        return await Runner.run(
            starting_agent=agent, input=input_data, run_config=config
        )


# --- Parallel Scoring Utility ---
async def run_parallel_scoring(
    domain: str,
    context_text: str,
) -> tuple[
    Optional[ConfidenceScoreSchema],
    Optional[RelevanceScoreSchema],
    Optional[ClarityScoreSchema],
]:
    """Run scoring agents concurrently on the provided domain and context.

    Args:
        domain: The domain string being evaluated.
        context_text: The full text context to supply to the scoring agents.

    Returns:
        A tuple containing the confidence, relevance, and clarity score schemas.
        Any element may be ``None`` if the corresponding agent failed or
        produced invalid output.
    """

    combined_input = f"Domain: {domain}\n\n{context_text}"

    score_tasks = [
        run_agent_with_retry(confidence_score_agent, combined_input),
        run_agent_with_retry(relevance_score_agent, combined_input),
        run_agent_with_retry(clarity_score_agent, combined_input),
    ]

    results = await asyncio.gather(*score_tasks, return_exceptions=True)

    confidence_data: Optional[ConfidenceScoreSchema] = None
    relevance_data: Optional[RelevanceScoreSchema] = None
    clarity_data: Optional[ClarityScoreSchema] = None

    # Confidence score processing
    conf_result = results[0]
    if isinstance(conf_result, Exception):
        logger.error("Confidence scoring failed", exc_info=conf_result)
    else:
        potential_output = getattr(conf_result, "final_output", None)
        if isinstance(potential_output, ConfidenceScoreSchema):
            confidence_data = potential_output
        elif isinstance(potential_output, dict):
            try:
                confidence_data = ConfidenceScoreSchema.model_validate(potential_output)
            except ValidationError as e:
                logger.warning("ConfidenceScoreSchema validation error: %s", e)
        else:
            logger.error(
                "Unexpected confidence score output type: %s",
                type(potential_output),
            )

    # Relevance score processing
    rel_result = results[1]
    if isinstance(rel_result, Exception):
        logger.error("Relevance scoring failed", exc_info=rel_result)
    else:
        potential_output = getattr(rel_result, "final_output", None)
        if isinstance(potential_output, RelevanceScoreSchema):
            relevance_data = potential_output
        elif isinstance(potential_output, dict):
            try:
                relevance_data = RelevanceScoreSchema.model_validate(potential_output)
            except ValidationError as e:
                logger.warning("RelevanceScoreSchema validation error: %s", e)
        else:
            logger.error(
                "Unexpected relevance score output type: %s",
                type(potential_output),
            )

    # Clarity score processing
    clar_result = results[2]
    if isinstance(clar_result, Exception):
        logger.error("Clarity scoring failed", exc_info=clar_result)
    else:
        potential_output = getattr(clar_result, "final_output", None)
        if isinstance(potential_output, ClarityScoreSchema):
            clarity_data = potential_output
        elif isinstance(potential_output, dict):
            try:
                clarity_data = ClarityScoreSchema.model_validate(potential_output)
            except ValidationError as e:
                logger.warning("ClarityScoreSchema validation error: %s", e)
        else:
            logger.error(
                "Unexpected clarity score output type: %s",
                type(potential_output),
            )

    return confidence_data, relevance_data, clarity_data


async def score_sub_domains(
    sub_domain_data: SubDomainSchema, context_text: str
) -> SubDomainSchema:
    """Score each sub-domain within ``sub_domain_data``.

    Each ``SubDomainDetail`` item will receive confidence, relevance,
    and clarity scores calculated via :func:`run_parallel_scoring`.

    Parameters
    ----------
    sub_domain_data:
        The initial sub-domain analysis output.
    context_text:
        The original text content used for scoring.

    Returns
    -------
    SubDomainSchema
        The updated schema with scores populated on each sub-domain.
    """

    tasks = [
        run_parallel_scoring(item.sub_domain, context_text)
        for item in sub_domain_data.identified_sub_domains
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for item, result in zip(sub_domain_data.identified_sub_domains, results):
        if isinstance(result, Exception):
            logger.error(
                "Scoring failed for sub-domain '%s'", item.sub_domain, exc_info=result
            )
            continue

        conf_data, rel_data, clar_data = cast(
            tuple[
                Optional[ConfidenceScoreSchema],
                Optional[RelevanceScoreSchema],
                Optional[ClarityScoreSchema],
            ],
            result,
        )
        item.confidence_score = conf_data.confidence_score if conf_data else None
        item.relevance_score = rel_data.relevance_score if rel_data else None
        item.clarity_score = clar_data.clarity_score if clar_data else None

    return sub_domain_data


async def score_topics(topic_data: TopicSchema, context_text: str) -> TopicSchema:
    """Score each topic within ``topic_data``.

    Each :class:`TopicDetail` item will receive confidence, relevance,
    and clarity scores calculated via :func:`run_parallel_scoring`.

    Parameters
    ----------
    topic_data:
        The aggregated topic analysis output.
    context_text:
        The original text content used for scoring.

    Returns
    -------
    TopicSchema
        The updated schema with scores populated on each topic.
    """

    tasks = []
    topic_items: List[TopicDetail] = []

    for sub_domain in topic_data.sub_domain_topic_map:
        for topic in sub_domain.identified_topics:
            tasks.append(run_parallel_scoring(topic.topic, context_text))
            topic_items.append(topic)

    if not tasks:
        return topic_data

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for item, result in zip(topic_items, results):
        if isinstance(result, Exception):
            logger.error("Scoring failed for topic '%s'", item.topic, exc_info=result)
            continue

        conf_data, rel_data, clar_data = cast(
            tuple[
                Optional[ConfidenceScoreSchema],
                Optional[RelevanceScoreSchema],
                Optional[ClarityScoreSchema],
            ],
            result,
        )
        item.confidence_score = conf_data.confidence_score if conf_data else None
        item.relevance_score = rel_data.relevance_score if rel_data else None
        item.clarity_score = clar_data.clarity_score if clar_data else None

    return topic_data
