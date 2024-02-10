# This is workable for token classification

import numpy as np


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }



# General purpose evaluation function
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_classification(predictions, labels):
    """
    Evaluate classification predictions using accuracy, precision, recall, and F1 score.
    
    Args:
        predictions (list): Predicted labels.
        labels (list): True labels.
    
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1 score.
    """
    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    # Return results as a dictionary
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Example usage:
# Assuming 'predictions' and 'labels' are your predicted and true labels
evaluation_results = evaluate_classification(predictions, labels)
print("Evaluation results:", evaluation_results)


# 
