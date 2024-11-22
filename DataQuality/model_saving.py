import json
import os


def save_top_performances(top_performances, TOP_PERFORMANCE_FILE):
    with open(TOP_PERFORMANCE_FILE, 'w') as f:
        json.dump(top_performances, f, indent=4)

def load_top_performances(TOP_PERFORMANCE_FILE):
    if os.path.exists(TOP_PERFORMANCE_FILE):
        with open(TOP_PERFORMANCE_FILE, 'r') as f:
            return json.load(f)
    return []


def update_top_performances(top_performances, accuracy, model_name, report, TOP_K, MODEL_SAVE_PATH, TOP_PERFORMANCE_FILE):
    if len(top_performances) < TOP_K or accuracy > min([p["accuracy"] for p in top_performances]):
        # Add the new performance and sort the list
        top_performances.append({"accuracy": accuracy, "model_name": model_name, "report": report})
        top_performances = sorted(top_performances, key=lambda x: x["accuracy"], reverse=True)
        # Keep only the top 10
        if len(top_performances) > TOP_K:
            # Remove the worst performance and delete the associated model file
            worst_performance = top_performances.pop()
            model_to_delete = os.path.join(MODEL_SAVE_PATH, worst_performance["model_name"])
            if os.path.exists(model_to_delete):
                os.remove(model_to_delete)
        # Save updated list
        save_top_performances(top_performances, TOP_PERFORMANCE_FILE)
    else:
        # If not top 10, delete the current model
        model_to_delete = os.path.join(MODEL_SAVE_PATH, model_name)
        if os.path.exists(model_to_delete):
            os.remove(model_to_delete)