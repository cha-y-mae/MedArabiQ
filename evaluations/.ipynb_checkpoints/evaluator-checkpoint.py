import json
from evaluations.metrics import calculate_accuracy, calculate_bleu, calculate_rouge, calculate_bert_score

def evaluate(predictions_path, metrics_path, task_type):
    """
    Evaluate model predictions using task-specific metrics.
    """
    import pandas as pd

    #load predictions
    # Load predictions
    predictions_df = pd.read_csv(predictions_path, dtype=str)  
    predictions = predictions_df['prediction'].astype(str).tolist()  
    ground_truths = predictions_df['ground_truth'].astype(str).tolist()  


    metrics = {}

    if task_type == "qa":
        metrics['accuracy'] = calculate_accuracy(predictions, ground_truths)
    #elif task_type == "summarization":
    #    metrics['bleu'] = calculate_bleu(predictions, ground_truths)

    elif task_type == "fib_closed":
        metrics['accuracy'] = calculate_accuracy(predictions, ground_truths)

    elif task_type == "fib_open":
        metrics['bleu'] = calculate_bleu(predictions, ground_truths)
        metrics.update(calculate_rouge(predictions, ground_truths))
        metrics.update(calculate_bert_score(predictions, ground_truths))

    elif task_type == "aramed":
        metrics['bleu'] = calculate_bleu(predictions, ground_truths)
        metrics.update(calculate_rouge(predictions, ground_truths))
        metrics.update(calculate_bert_score(predictions, ground_truths))
        
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics

if __name__ == "__main__":
    import sys
    predictions_file = sys.argv[1]  
    metrics_file = sys.argv[2]      
    task = sys.argv[3]              

    metrics = evaluate(predictions_file, metrics_file, task)
    print("Evaluation completed. Metrics saved to:", metrics_file)
