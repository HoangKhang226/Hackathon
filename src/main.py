from agents import *
from pipeline import *
from core import load_model


def get_batches(state, batch_size: int = 10):
    questions = state.get("questions", [])
    final_answers = state.get("final_answers", [])

    pending_items = []
    for idx, question in enumerate(questions):
        if idx >= len(final_answers) or final_answers[idx] in (None, ""):
            item = dict(question)
            item["idx"] = idx
            pending_items.append(item)

    for start in range(0, len(pending_items), batch_size):
        yield pending_items[start:start + batch_size]



def main():
    # 1. Load model (Khải)
    engine = load_model()
    
    # 2. Load data
    df = load_test_data()
    state = load_checkpoint() or init_graph_state(df)
    
    # 3. Xử lý theo batch
    for batch in get_batches(state, batch_size=10):
        routes = llm_router_node(engine, batch)
        
        # Phân nhánh
        for item, route in zip(batch, routes):
            if route == "FAST_QA":
                answer = fast_qa_agent_node(engine, item)
            elif route == "READING":
                answer = reading_comprehension_agent_node(engine, item)
            elif route == "CODEABLE":
                answer = coder_node(engine, item)  # Huy
            
            # Lọc rác
            final = map_answer(answer["final_answer"], item["choices"], state["choice_counts"][item["idx"]])
            state["final_answers"][item["idx"]] = final
        
        # Lưu checkpoint
        save_checkpoint(state)
    
    # 4. Ghi kết quả
    save_predictions(state)

if __name__ == "__main__":
    main()