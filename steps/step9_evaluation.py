"""Step 9: Evaluation and filtering of verified extractions."""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, List, Optional

from pydantic import ValidationError

try:
    from agentic_team import RunConfig, RunResult, TResponseInputItem
except ImportError:
    print("Error: 'agentic_team' SDK library not found or incomplete for step 9.")
    raise

from ..agents import evaluator_agent
from ..config import EVALUATION_MODEL, EVALUATION_OUTPUT_DIR, EVALUATION_OUTPUT_FILENAME
from ..schemas import EvaluationResultSchema
from ..utils import direct_save_json_output, run_agent_with_retry

logger = logging.getLogger(__name__)

async def evaluate_verified_extractions(
    verified_extractions: Any,
    overall_trace_id: Optional[str] = None
) -> Optional[EvaluationResultSchema]:
    """Evaluate verified extractions and filter final enrichment triples."""
    if not verified_extractions:
        logger.info("Skipping Step 9 because no verified extractions were provided.")
        return None

    logger.info(f"--- Running Step 9: Evaluation (Agent: {evaluator_agent.name}) ---")
    print(f"\n--- Running Step 9: Evaluation using model: {EVALUATION_MODEL} ---")

    step9_metadata = {
        "workflow_step": "9_evaluation",
        "agent_name": "Evaluation",
        "actual_agent": str(evaluator_agent.name),
    }
    step9_run_config = RunConfig(trace_metadata={k: str(v) for k, v in step9_metadata.items()})

    step9_input: List[TResponseInputItem] = [
        {
            "role": "user",
            "content": (
                "Evaluate the following verified extractions and return the final set of enrichment triples. "
                "Output only using the EvaluationResultSchema."
            ),
        },
        {"role": "user", "content": json.dumps(verified_extractions, indent=2)},
    ]

    try:
        step9_result: Optional[RunResult] = await run_agent_with_retry(
            agent=evaluator_agent,
            input_data=step9_input,
            config=step9_run_config,
        )
    except Exception as e:
        logger.exception("Error running EvaluatorAgent", extra={"trace_id": overall_trace_id or 'N/A'})
        print(f"Error running EvaluatorAgent: {type(e).__name__}: {e}")
        return None

    evaluation_data: Optional[EvaluationResultSchema] = None
    if step9_result:
        potential_output = getattr(step9_result, "final_output", None)
        if isinstance(potential_output, EvaluationResultSchema):
            evaluation_data = potential_output
        elif isinstance(potential_output, dict):
            try:
                evaluation_data = EvaluationResultSchema.model_validate(potential_output)
            except ValidationError as e:
                logger.warning(f"Dict output failed EvaluationResultSchema validation: {e}")
        else:
            logger.warning(f"Unexpected output type from EvaluatorAgent: {type(potential_output)}")

    if evaluation_data:
        output_content = {
            "final_triples": [t.model_dump() for t in evaluation_data.final_triples],
            "evaluation_summary": evaluation_data.evaluation_summary,
            "analysis_details": {
                "model_used": EVALUATION_MODEL,
                "agent_name": evaluator_agent.name,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            "trace_information": {"trace_id": overall_trace_id or "N/A"},
        }
        save_result = direct_save_json_output(
            EVALUATION_OUTPUT_DIR,
            EVALUATION_OUTPUT_FILENAME,
            output_content,
            overall_trace_id,
        )
        print(f"\nSaving evaluation output file...\n  - {save_result}")
        logger.info(f"Result of saving evaluation output: {save_result}")

    return evaluation_data
