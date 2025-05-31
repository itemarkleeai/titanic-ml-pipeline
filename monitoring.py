# Example monitoring script for model drift/logging

def log_accuracy(run_id, accuracy):
    with open("model_accuracy_log.csv", "a") as f:
        f.write(f"{run_id},{accuracy}\n")

# Example usage
log_accuracy("2025-06-01_rf_v1", 0.81)
