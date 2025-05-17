import asyncio
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

# Import components from the workflow package
try:
    from utils import (
        parse_arguments,
        read_input_from_file,
        read_input_from_directory,
        prompt_user_for_input,
        setup_logging,
    )
    from orchestrator import run_combined_workflow
    from steps import generate_workflow_visualization
    from config import (
        PYMUPDF_AVAILABLE,
        MAX_INPUT_CONTENT_LENGTH,
    )  # For initial logging
except ImportError as e:
    print(
        "FATAL ERROR: Could not import necessary components from 'graphyte_ai'. Ensure the package is installed correctly and accessible."
    )
    print(f"Import Error: {e}")
    # Try to provide a hint if it's a module not found related to the package itself
    if "graphyte_ai" in str(e):
        print(
            "Hint: Make sure you are running this script from the project root directory or that 'graphyte_ai' is in your Python path."
        )
    sys.exit(1)

# Setup logging as early as possible
setup_logging()
logger = logging.getLogger(__name__)  # Get logger for the main script


# --- Async Main Function ---
async def main_async() -> None:
    """Main asynchronous function to parse arguments, read input, and orchestrate the agent workflow."""
    logger.info("=== Starting Document Analysis Workflow ===")
    try:
        # Optional: Timezone context logging
        import pytz  # type: ignore

        tz = pytz.timezone("Etc/UTC")  # Or try to get local timezone
        logger.info(f"Using timezone context: {tz.zone}")
    except ImportError:
        # Fallback if pytz is not installed
        # Use the timezone already imported at the top of the file
        tz = timezone.utc
        logger.warning("pytz library not found. Using UTC for timezone context.")

    current_time = datetime.now(tz)
    logger.info(f"Workflow start time: {current_time.isoformat()}")

    args = parse_arguments()
    logger.debug(f"Parsed arguments: {args}")

    # --- Handle Visualization Request ---
    if args.visualize:
        logger.info("Visualization flag detected. Running visualization generation.")
        await generate_workflow_visualization()
        logger.info("Visualization generation complete. Exiting.")
        sys.exit(0)  # Exit after generating visualization

    # --- Proceed with Normal Workflow if not visualizing ---
    content = ""

    input_source = "stdin"
    try:
        if args.file:
            input_source = f"file: {args.file}"
            # Resolve path here for better error messages if it doesn't exist
            file_arg_path = Path(args.file).resolve()
            content = read_input_from_file(file_arg_path)
        elif args.dir:
            input_source = f"dir: {args.dir}"
            dir_arg_path = Path(args.dir).resolve()
            content = read_input_from_directory(dir_arg_path)
        else:
            content = prompt_user_for_input()

        # Check content length warning
        if len(content) > MAX_INPUT_CONTENT_LENGTH:
            logger.warning(
                f"Input content length ({len(content)} characters) exceeds threshold ({MAX_INPUT_CONTENT_LENGTH}). May impact performance/cost."
            )
            print(
                f"\nWarning: Input content length ({len(content):,} characters) is large and may result in long processing times or high costs."
            )

        # Proceed only if content is valid (and we are not visualizing)
        if content and content.strip():
            logger.info(
                f"Obtained content from {input_source} (length: {len(content)}). Running analysis workflow."
            )
            await run_combined_workflow(content)

        else:
            # Log and print specific messages based on the input source if no content was found
            if input_source == "stdin":
                logger.warning(
                    "No input content provided via stdin or input was cancelled."
                )
                print("No input content provided. Exiting.")
            elif args.dir:
                logger.warning(
                    f"Directory '{args.dir}' provided, but no readable text content found or processed."
                )
                print(
                    f"Directory '{args.dir}' provided, but no readable text content found or processed. Exiting."
                )
            elif args.file:
                logger.warning(f"File '{args.file}' resulted in empty content.")
                print(f"File '{args.file}' resulted in empty content. Exiting.")
            else:  # Should not happen if args parsing is correct, but include for completeness
                logger.warning("No valid input source provided or content was empty.")
                print("No input content found. Exiting.")

    # Catch specific input errors
    except (
        FileNotFoundError,
        NotADirectoryError,
        IsADirectoryError,
        IOError,
        OSError,
        ImportError,
    ) as e:
        logger.exception(
            f"Input Error processing source '{input_source}': {type(e).__name__}: {e}"
        )
        print(
            f"\nError processing input source '{input_source}': {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        if (
            isinstance(e, ImportError)
            and PYMUPDF_AVAILABLE is False
            and (".pdf" in input_source if isinstance(input_source, str) else False)
        ):
            print(
                "Hint: PDF processing failed. Ensure 'PyMuPDF' is installed ('pip install pymupdf').",
                file=sys.stderr,
            )
        sys.exit(1)
    # Catch unexpected errors during input processing or workflow execution
    except Exception as e:
        logger.exception(
            f"Unhandled exception during processing from '{input_source}'."
        )
        print(
            f"\nAn unexpected error occurred: {type(e).__name__}: {e}", file=sys.stderr
        )
        sys.exit(1)
    finally:
        logger.info("=== Document Analysis Workflow Finished ===")


# --- Entry Point ---
if __name__ == "__main__":
    # Log initial status using the configured logger
    logger.info(
        f"Starting main application coroutine using Python {sys.version_info.major}.{sys.version_info.minor}."
    )
    # Optionally log status of dependencies again here if useful
    # if not PYMUPDF_AVAILABLE: logger.warning(...)
    # if not TENACITY_AVAILABLE: logger.warning(...)

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nExecution cancelled by user (KeyboardInterrupt).")
        logger.warning("Execution cancelled by user (KeyboardInterrupt).")
        sys.exit(0)
    except Exception as e:
        # Catch any truly fatal errors not caught within main_async
        logger.critical(
            f"Fatal error running the application: {type(e).__name__}: {e}",
            exc_info=True,
        )
        print(
            f"\nFATAL ERROR: Failed to run the application: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
