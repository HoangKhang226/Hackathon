import os
import pandas as pd
from src.core.llm_engine import load_model
from src.pipeline.io_handler import load_test_data, save_predictions
from src.pipeline.checkpointing import load_checkpoint, save_checkpoint
from src.pipeline.dynamic_mapper import map_answer
from src.agents.state import init_graph_state
from src.agents.router import llm_router_node
from src.agents.fast_qa import fast_qa_agent_node
from src.agents.reading import reading_comprehension_agent_node
from src.agents.coder import coder_node
from src.pipeline.majority_voting import run_majority_voting
from src.utils.logger import logger

def main():
    logger.info("Starting HackAIthon 2026 Pipeline...")
    
    # 1. Load model & tokenizer
    model, tokenizer = load_model()
    
    # 2. Load data and initialize state
    try:
        df = load_test_data()
        logger.info(f"Loaded test data with {len(df)} questions.")
        
        # Parse choices correctly if they are strings (like JSON strings or just letters)
        # Assuming the CSV has a 'choices' column which is a list or string representation of a list
        import ast
        questions_list = df.to_dict("records")
        for q in questions_list:
            if "choices" in q and isinstance(q["choices"], str):
                try:
                    q["choices"] = ast.literal_eval(q["choices"])
                except:
                    # If it fails, maybe split by some delimiter or keep as is if already a list
                    pass
    except FileNotFoundError:
        logger.error("Test data not found in /data. Ensure files are mounted correctly.")
        return

    state = load_checkpoint()
    if state is None:
        state = init_graph_state(questions_list)
        logger.info("Initialized fresh GraphState.")
    else:
        logger.info("Resumed from checkpoint.")
    
    # 3. Graph Execution
    # Node 1: Router
    state = llm_router_node(state, model, tokenizer, checkpoint_callback=save_checkpoint)
    
    # Node 2 & 3: Direct Answering
    state = fast_qa_agent_node(state, model, tokenizer, checkpoint_callback=save_checkpoint)
    state = reading_comprehension_agent_node(state, model, tokenizer, checkpoint_callback=save_checkpoint)
    
    # Node 4 & 5: Coding and Sandbox
    state = coder_node(state, model, tokenizer)
    codeable_indices = [i for i, r in enumerate(state["routes"]) if r == "CODEABLE"]
    state = run_majority_voting(state, model, tokenizer, codeable_indices)
    
    # 4. Filter and Dynamic Mapping
    logger.info("Mapping final answers using Dynamic Mapper...")
    results = []
    for idx, (q, raw_ans) in enumerate(zip(state["questions"], state["final_answers"])):
        num_choices = state["choice_counts"][idx]
        choices = q.get("choices", [])
        
        if not raw_ans:
            final_ans = "A" # Fallback if empty
        else:
            final_ans = map_answer(raw_ans, choices, num_choices)
            
        qid = q.get("qid") if pd.notna(q.get("qid")) else q.get("id", str(idx))
        results.append({"qid": qid, "answer": final_ans})
        
    # 5. Save Predictions
    logger.info("Saving predictions to /output/pred.csv...")
    save_predictions(results)
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()