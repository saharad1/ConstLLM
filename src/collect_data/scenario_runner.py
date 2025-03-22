"""
Runner for executing scenarios in the data collection pipeline.

This module handles the execution flow for scenarios, including retry logic,
error handling, and saving results to files. It acts as a wrapper around the
core processing functionality, providing robustness and persistence.
"""
import time
import traceback
from dataclasses import asdict

# Import the process_scenario function from the renamed module
from src.collect_data.scenario_core_processor import process_scenario, get_scenario_attribute


def process_single_scenario(scenario_item, llm_analyzer, methods_params_decision, 
                           methods_params_explanation, num_dec_exp, original_params, 
                           iteration, max_retries=3):
    """
    Process a single scenario with retries.
    
    Args:
        scenario_item: The scenario item to process
        llm_analyzer: The LLM analyzer instance
        methods_params_decision: Parameters for decision attribution methods
        methods_params_explanation: Parameters for explanation attribution methods
        num_dec_exp: Number of decision explanations to generate
        original_params: Original parameters for the scenario
        iteration: Current iteration number
        max_retries: Maximum number of retries for processing a scenario
        
    Returns:
        Tuple of (scenario_result, error_message)
    """
    # Extract scenario ID for tracking
    scenario_id = get_scenario_attribute(scenario_item, "scenario_id", "scenario_id", f"scenario_{iteration}")
    
    # Initialize error message
    error_msg = None
    
    # Try processing with retries
    for retry in range(max_retries):
        try:
            # Process the scenario using the process_scenario function
            scenario_res = process_scenario(
                llm_analyzer=llm_analyzer,
                scenario_item=scenario_item,
                methods_params_decision=methods_params_decision,
                methods_params_explanation=methods_params_explanation,
                num_dec_exp=num_dec_exp,
                custom_logger=None,  # We'll use the default logger
            )
            
            # Ensure scenario_id is set
            if not hasattr(scenario_res, "scenario_id") or not scenario_res.scenario_id:
                scenario_res.scenario_id = scenario_id
                
            return scenario_res, None
            
        except Exception as e:
            error_msg = f"Error processing scenario (retry {retry+1}/{max_retries}): {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            
            # Wait before retrying
            time.sleep(2)
            
            # Reset parameters for retry
            methods_params_decision = original_params["decision"].copy()
            methods_params_explanation = original_params["explanation"].copy()
    
    # If we get here, all retries failed
    return None, error_msg


def save_scenario_result(scenario_res, jsonl_filename):
    """
    Save a scenario result to a JSONL file.
    
    Args:
        scenario_res: The scenario result to save
        jsonl_filename: Path to the JSONL file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert dataclass to dictionary if needed
        scenario_dict = asdict(scenario_res) if hasattr(scenario_res, "__dataclass_fields__") else scenario_res
        
        # Write to JSONL file
        with open(jsonl_filename, "a") as f:
            f.write(f"{scenario_dict}\n")
        return True
    except Exception as e:
        print(f"Error saving scenario result: {e}")
        return False
