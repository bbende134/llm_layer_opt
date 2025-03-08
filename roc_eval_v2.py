import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import scienceplots
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import torch
import evaluate
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

# Set up SciencePlots style
plt.style.context(['science', 'nature'])

# Function to load and tokenize dataset
def load_validation_data(file_path):
    # df = pd.read_csv(file_path)
    df = pd.read_csv(file_path).drop(columns=['class_name']).dropna().drop_duplicates()
    df['text'] = df['text'].apply(lambda x: x.replace('\n', ''))
    # Assuming the validation data has 'text' and 'label' columns
    # If your data has different column names, adjust accordingly
    text_column = 'text' if 'text' in df.columns else df.columns[0]
    label_column = 'label' if 'label' in df.columns else df.columns[1]
    
    return df[text_column].tolist(), df[label_column].tolist()

def get_model_predictions(model, tokenizer, texts, device):
    model.eval()
    all_probs = []
    
    # Process in batches to avoid memory issues
    batch_size = 16
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits = outputs.logits
        # Perform softmax on GPU before transferring to CPU
        probs = torch.softmax(logits, dim=1)
        # Move to CPU and convert to numpy only after computation
        probs = probs.cpu().numpy()
        all_probs.extend(probs)
    
    return np.array(all_probs)

# Function to compute ROC curves and AUC for each class (OvO approach)
def compute_multiclass_roc_ovo(y_true, y_pred_proba, n_classes):
    # Dictionary to store ROC curve data for each class pair
    roc_data = {}
    
    # Calculate ROC AUC using sklearn for multiclass (OvO) - more reliable in this case
    # Convert labels to proper format for sklearn
    y_true_array = np.array(y_true)
    
    # Calculate macro and micro average AUC using sklearn
    macro_auc = roc_auc_score(
        label_binarize(y_true_array, classes=range(n_classes)), 
        y_pred_proba, 
        multi_class='ovo', 
        average='macro'
    )
    
    micro_auc = roc_auc_score(
        label_binarize(y_true_array, classes=range(n_classes)), 
        y_pred_proba, 
        multi_class='ovo', 
        average='micro'
    )
    
    # Get class names or indices
    class_names = [f"Class {i}" for i in range(n_classes)]
    
    # Compute ROC curve for each class pair
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            # Get samples belonging to classes i and j
            mask = np.where(np.isin(y_true_array, [i, j]))[0]
            if len(mask) == 0:
                continue
                
            # Subset data
            y_bin_ij = np.array([1 if y_true_array[k] == i else 0 for k in mask])
            y_score_ij = y_pred_proba[mask][:, i] / (y_pred_proba[mask][:, i] + y_pred_proba[mask][:, j])
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_bin_ij, y_score_ij)
            roc_auc = auc(fpr, tpr)
            
            # Store data
            roc_data[(class_names[i], class_names[j])] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            }
    
    return roc_data, micro_auc, macro_auc

# Function to plot ROC curves
def plot_multiclass_roc(roc_data, micro_auc, macro_auc, model_name, n_classes):
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(roc_data)))
    
    for i, ((class_i, class_j), values) in enumerate(roc_data.items()):
        plt.plot(
            values['fpr'], 
            values['tpr'], 
            lw=2,
            color=colors[i],
            label=f'{class_i} vs {class_j} (AUC = {values["auc"]:.2f})'
        )
    
    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multiclass ROC Curves (OvO) - {model_name}\nMicro-Avg AUC: {micro_auc:.3f}, Macro-Avg AUC: {macro_auc:.3f}')
    
    # Adjust legend placement to minimize overlap with curves
    plt.legend(loc='lower right', fontsize='small')
    
    plt.tight_layout()
    return plt.gcf()

# Helper function to run huggingface evaluate metric safely
def compute_roc_auc_safe(labels, predictions, multi_class="ovo", average="micro"):
    """
    Safely compute ROC AUC using huggingface evaluate or fallback to sklearn
    """
    try:
        # Try using huggingface evaluate
        roc_auc_metric = evaluate.load("evaluate-metric/roc_auc")
        
        # Create a dataset in the format expected by evaluate
        eval_dataset = Dataset.from_dict({
            "references": [int(l) for l in labels],  # Ensure integers
            "prediction_scores": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
        })
        
        result = roc_auc_metric.compute(
            references=eval_dataset["references"],
            prediction_scores=eval_dataset["prediction_scores"],
            multi_class=multi_class,
            average=average
        )
        return result
    except (ValueError, TypeError) as e:
        print(f"Warning: Huggingface evaluate failed with error: {e}")
        print("Falling back to sklearn for ROC AUC calculation")
        
        # Fallback to sklearn
        return roc_auc_score(
            label_binarize(labels, classes=range(predictions.shape[1])), 
            predictions, 
            multi_class=multi_class, 
            average=average
        )

# Main execution
def main():
    # Get all fine-tuned model directories
    model_dirs = glob.glob('fine_tuned_model*')
    
    if not model_dirs:
        print("No fine-tuned model directories found. Please check if they exist.")
        return
    
    # Load validation data
    val_path = 'data/validation.csv'
    if not os.path.exists(val_path):
        print(f"Validation data not found at {val_path}")
        return
    
    texts, labels = load_validation_data(val_path)
    
    # Determine device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine number of classes from validation data
    n_classes = len(set(labels))
    print(f"Detected {n_classes} classes in validation data")
    
    results_dir = 'eval/images'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # Process each model
    for i, model_dir in enumerate(model_dirs):
        print(f"\nEvaluating model: {model_dir}")
        
        # Load model and tokenizer
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
        except Exception as e:
            print(f"Error loading model {model_dir}: {e}")
            continue
        
        # Get model predictions
        predictions = get_model_predictions(model, tokenizer, texts, device)
        
        # Compute ROC curves and AUC
        roc_data, micro_auc, macro_auc = compute_multiclass_roc_ovo(labels, predictions, n_classes)
        
        # Plot ROC curves
        plot_fig = plot_multiclass_roc(roc_data, micro_auc, macro_auc, os.path.basename(model_dir), n_classes)
        

        # Save individual model plot
        plt.savefig(f'eval/images/roc_curve_{os.path.basename(model_dir)}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'eval/images/roc_curve_{os.path.basename(model_dir)}.png', dpi=300, bbox_inches='tight')
    
    # Create a summary plot comparing all models
    plt.figure(figsize=(10, 8))
    plt.style.use(['science', 'grid'])
    
    # Load and compare all models with their micro-average ROC AUC
    all_models_results = []
    
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            predictions = get_model_predictions(model, tokenizer, texts, device)
            
            # Use sklearn for consistent results
            micro_auc = roc_auc_score(
                label_binarize(labels, classes=range(n_classes)), 
                predictions, 
                multi_class='ovo', 
                average='micro'
            )
            
            all_models_results.append((model_name, micro_auc))
        except Exception as e:
            print(f"Error processing model {model_dir} for summary: {e}")
    
    # Sort by performance
    all_models_results.sort(key=lambda x: x[1], reverse=True)
    
    # Plot summary
    plt.barh([r[0] for r in all_models_results], [r[1] for r in all_models_results])
    plt.xlabel('Micro-Average ROC AUC (OvO)')
    plt.title('Model Comparison - Multiclass ROC AUC')
    plt.tight_layout()
    results_dir = 'eval/combined'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    # Save summary plot
    plt.savefig('eval/combined/model_comparison_roc_auc.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('eval/combined/model_comparison_roc_auc.png', dpi=300, bbox_inches='tight')
    
    print("\nEvaluation completed. Results saved as PDF and PNG files.")

if __name__ == "__main__":
    main()