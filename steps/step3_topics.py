"""Step 3: Topic identification functionality."""

import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from pydantic import ValidationError

from agentic_team import RunConfig, RunResult, TResponseInputItem

from ..agents import topic_identifier_agent
from ..config import TOPIC_MODEL, TOPIC_OUTPUT_DIR, TOPIC_OUTPUT_FILENAME
from ..schemas import TopicSchema, SingleSubDomainTopicSchema, SubDomainSchema
from ..utils import direct_save_json_output, run_agent_with_retry

logger = logging.getLogger(__name__)

async def identify_topics(
    content: str, 
    primary_domain: str, 
    sub_domain_data: SubDomainSchema, 
    overall_trace_id: Optional[str] = None
) -> Optional[TopicSchema]:
    """Identify topics for each sub-domain from the input content.
    
    Args:
        content: The text content to analyze
        primary_domain: The primary domain identified in step 1
        sub_domain_data: The SubDomainSchema from step 2
        overall_trace_id: The trace ID for logging purposes
        
    Returns:
        A TopicSchema object if successful, None otherwise
    """
    if not primary_domain or not sub_domain_data:
        logger.info("Skipping Step 3 because prerequisites (primary domain or sub-domains) were not identified.")
        print("Skipping Step 3 as primary domain or sub-domains were not identified.")
        return None
        
    sub_domains_list_for_step3 = [
        item.sub_domain.strip() for item in sub_domain_data.identified_sub_domains 
        if item.sub_domain and item.sub_domain.strip()
    ]
    
    if not sub_domains_list_for_step3:
        logger.info("Skipping Step 3 because no valid sub-domains were identified.")
        print("Skipping Step 3 as no valid sub-domains were identified.")
        return None
        
    logger.info(f"--- Starting Step 3: PARALLEL Topic ID (Agent: {topic_identifier_agent.name}) for {len(sub_domains_list_for_step3)} Sub-Domain(s) ---")
    print(f"\n--- Running Step 3: PARALLEL Topic ID using model: {TOPIC_MODEL} ---")

    topic_tasks = []
    sub_domains_being_processed = [] # Keep track of which sub-domain corresponds to which task/result
    aggregated_topic_results: List[SingleSubDomainTopicSchema] = []
    final_topic_data: Optional[TopicSchema] = None

    # --- Prepare tasks for parallel execution ---
    for index, current_sub_domain in enumerate(sub_domains_list_for_step3):
        if not current_sub_domain: # Should have been filtered, but double-check
            logger.warning(f"Skipping empty sub-domain string at index {index} during task preparation for Step 3.")
            continue

        logger.debug(f"Preparing task for Step 3 ({index+1}/{len(sub_domains_list_for_step3)}): Sub-Domain '{current_sub_domain}'")

        display_sub_domain = (current_sub_domain[:25] + '...') if len(current_sub_domain) > 28 else current_sub_domain
        step3_iter_metadata_for_trace = {
            "workflow_step": f"3_topic_id_batch_{index+1}",
            "agent_name": f"Topic ID ({display_sub_domain})",
            "actual_agent": str(topic_identifier_agent.name),
            "primary_domain_input": primary_domain,
            "sub_domain_analyzed": current_sub_domain,
            "batch_index": str(index+1),
            "batch_size": str(len(sub_domains_list_for_step3)),
        }
        step3_iter_run_config = RunConfig(trace_metadata={k: str(v) for k, v in step3_iter_metadata_for_trace.items()})

        step3_iter_input_list: List[TResponseInputItem] = [
            {"role": "user", "content": f"The primary domain is '{primary_domain}'. Focus ONLY on the sub-domain: '{current_sub_domain}'. Based ONLY on the following text, identify specific topics mentioned within the text relevant ONLY to this sub-domain ('{current_sub_domain}'). Output using the required SingleSubDomainTopicSchema."},
            {"role": "user", "content": f"--- Full Text Start ---\n{content}\n--- Full Text End ---"}
        ]

        # Create the async task using the retry wrapper
        task = asyncio.create_task(
            run_agent_with_retry(
                agent=topic_identifier_agent,
                input_data=step3_iter_input_list,
                config=step3_iter_run_config
            ),
            name=f"TopicTask_{current_sub_domain[:20]}" # Optional: name task for debugging
        )
        topic_tasks.append(task)
        sub_domains_being_processed.append(current_sub_domain) # Track the sub-domain for this task

    # --- Execute tasks in parallel ---
    if not topic_tasks:
        logger.warning("No valid sub-domains found to process in Step 3. Skipping parallel execution and subsequent steps.")
        print("No valid sub-domains to process for topics. Cannot proceed further.")
        return None
        
    logger.info(f"Launching {len(topic_tasks)} topic identification tasks in parallel...")
    print(f"Running topic identification for {len(topic_tasks)} sub-domains concurrently...")
    
    # Use asyncio.gather to run tasks concurrently and collect results/exceptions
    step3_results_list = await asyncio.gather(*topic_tasks, return_exceptions=True)
    logger.info("Parallel topic identification tasks completed.")
    print("Parallel topic identification runs finished. Processing results...")

    # --- Process results from parallel execution ---
    for index, step3_iter_result_or_exc in enumerate(step3_results_list):
        current_sub_domain = sub_domains_being_processed[index] # Get corresponding sub-domain

        try:
            # Check if an exception was returned by gather
            if isinstance(step3_iter_result_or_exc, Exception):
                logger.error(f"Step 3 task for '{current_sub_domain}' failed with exception: {step3_iter_result_or_exc}", exc_info=step3_iter_result_or_exc)
                print(f"  - Error processing sub-domain '{current_sub_domain}': {type(step3_iter_result_or_exc).__name__}: {step3_iter_result_or_exc}")
                continue # Skip to the next result

            # If no exception, it should be a RunResult (or None)
            step3_iter_result: Optional[RunResult] = step3_iter_result_or_exc

            if step3_iter_result:
                potential_output_iter = getattr(step3_iter_result, 'final_output', None)
                single_topic_data: Optional[SingleSubDomainTopicSchema] = None

                if isinstance(potential_output_iter, SingleSubDomainTopicSchema):
                    single_topic_data = potential_output_iter
                    logger.info(f"Successfully extracted SingleSubDomainTopicSchema for '{current_sub_domain}'.")
                elif isinstance(potential_output_iter, dict):
                    try:
                        single_topic_data = SingleSubDomainTopicSchema.model_validate(potential_output_iter)
                        logger.info(f"Successfully validated SingleSubDomainTopicSchema from dict for '{current_sub_domain}'.")
                    except ValidationError as e:
                        logger.warning(f"Dict output for '{current_sub_domain}' failed SingleSubDomainTopicSchema validation: {e}")
                else:
                    logger.warning(f"Output for '{current_sub_domain}' was not SingleSubDomainTopicSchema or dict (type: {type(potential_output_iter)}). Raw: {potential_output_iter}")

                if single_topic_data:
                    # Ensure the sub_domain in the output matches the one requested
                    if single_topic_data.sub_domain.strip().lower() != current_sub_domain.strip().lower():
                        logger.warning(f"Sub-domain mismatch in output for '{current_sub_domain}'. Output had '{single_topic_data.sub_domain}'. Correcting to requested sub-domain.")
                        single_topic_data.sub_domain = current_sub_domain # Overwrite with the requested sub-domain

                    topic_names = [f"'{item.topic}'" for item in single_topic_data.identified_topics]
                    logger.info(f"Step 3 Result for '{current_sub_domain}': Identified Topics = [{', '.join(topic_names)}]")
                    print(f"\n  --- Topics for Sub-Domain: '{current_sub_domain}' ---")
                    if topic_names:
                        for item in single_topic_data.identified_topics:
                            print(f"     - {item.topic}")
                    else:
                        print("     - (No specific topics identified for this sub-domain)")
                    # Add the successfully processed result to the list
                    aggregated_topic_results.append(single_topic_data)
                else:
                    logger.warning(f"Could not extract valid topic data for sub-domain '{current_sub_domain}'. Raw output: {potential_output_iter}")
                    print(f"  - Warning: Failed to get structured topics for '{current_sub_domain}'.")
            else:
                logger.error(f"Step 3 task for '{current_sub_domain}' returned no result object (result was None or Falsy).")
                print(f"  - Error: Failed to get result object for sub-domain '{current_sub_domain}'.")

        except (ValidationError, TypeError) as e:
            logger.exception(f"Validation or Type error processing result for '{current_sub_domain}'. Error: {e}", extra={"trace_id": overall_trace_id or 'N/A'})
            print(f"\nError: A data validation or type issue occurred processing result for sub-domain '{current_sub_domain}'.")
            print(f"Error details: {e}")
        except Exception as e:
            logger.exception(f"An unexpected error occurred processing result for '{current_sub_domain}'.", extra={"trace_id": overall_trace_id or 'N/A'})
            print(f"\nAn unexpected error occurred processing result for sub-domain '{current_sub_domain}': {type(e).__name__}: {e}")
    # --- End of processing loop for parallel results ---

    # === After Parallel Runs: Aggregate and Save Final Topic Output ===
    if not aggregated_topic_results:
        logger.warning("Step 3 (Parallel) completed, but no topic results were successfully aggregated. Final topic file not saved.")
        print("\nStep 3 (Parallel) completed, but no topic results were successfully aggregated. Skipping subsequent steps. Final topic file not saved.")
        return None
        
    logger.info("Aggregating topic results from parallel runs and saving final output.")
    print("\n--- Aggregating and Saving Final Topic Analysis (from parallel runs) ---")

    # Store the final aggregated data for Step 4
    final_topic_data = TopicSchema(
        primary_domain=primary_domain, # Use the confirmed primary domain from Step 1
        sub_domain_topic_map=aggregated_topic_results,
        analysis_summary=f"Generated topics in parallel for {len(aggregated_topic_results)} sub-domains (out of {len(sub_domains_being_processed)} attempted)." # Use processed count
    )

    logger.info(f"Final Aggregated Topics (Structured):\n{final_topic_data.model_dump_json(indent=2)}")
    print("\n--- Final Aggregated Topics (Structured Output from Step 3 Parallel Runs) ---")
    print(final_topic_data.model_dump_json(indent=2))

    topic_output_content = {
        "primary_domain": final_topic_data.primary_domain,
        "sub_domain_topic_map": [item.model_dump() for item in final_topic_data.sub_domain_topic_map],
        "analysis_summary": final_topic_data.analysis_summary,
        "analysis_details": {
            "source_text_length": len(content),
            "primary_domain_analyzed": primary_domain,
            "sub_domains_attempted": sub_domains_being_processed, # List of subdomains attempted
            "sub_domains_successfully_processed": [item.sub_domain for item in aggregated_topic_results],
            "sub_domain_input_source": "List extracted from Step 2 output (SubDomainSchema)",
            "execution_mode": "Parallel (asyncio.gather)",
            "model_used_per_topic_call": TOPIC_MODEL,
            "agent_name_per_topic_call": topic_identifier_agent.name,
            "output_schema_final": TopicSchema.__name__,
            "output_schema_per_call": SingleSubDomainTopicSchema.__name__,
            "timestamp_utc": datetime.now(timezone.utc).isoformat()
        },
        "trace_information": {
            "trace_id": overall_trace_id or "N/A",
            "notes": f"Aggregated from PARALLEL calls to {topic_identifier_agent.name} in Step 3 of workflow."
        }
    }
    save_result_step3_final = direct_save_json_output(
        TOPIC_OUTPUT_DIR, TOPIC_OUTPUT_FILENAME, topic_output_content, overall_trace_id
    )
    print(f"\nSaving final aggregated topic output file...")
    print(f"  - {save_result_step3_final}")
    logger.info(f"Result of saving final aggregated topic output: {save_result_step3_final}")
    
    return final_topic_data