import os
import glob
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import scienceplots
import torch
from evaluate import load
from itertools import combinations

# Set the scientific plotting style from SciencePlots
plt.style.context(['science'])

# Step 1: Load the validation dataset
validation_data = pd.read_csv("data/validation.csv").drop(columns=['class_name']).dropna().drop_duplicates()
validation_data['text'] = validation_data['text'].apply(lambda x: x.replace('\n', ''))
texts = validation_data["text"].tolist()
labels = validation_data["label"].tolist()

# Step 2: Identify model directories
model_dirs = glob.glob("fine_tuned_model*")
model_names = [os.path.basename(model_dir).replace("fine_tuned_model_", "") for model_dir in model_dirs]

# Step 3: Load tokenizer and model config from the first model
tokenizer = AutoTokenizer.from_pretrained(model_dirs[0])
first_model = AutoModelForSequenceClassification.from_pretrained(model_dirs[0])
num_classes = first_model.config.num_labels
id2label = getattr(first_model.config, 'id2label', None)
label2id = getattr(first_model.config, 'label2id', None)

# Step 4: Map true labels to indices if necessary
true_labels = np.array(labels)
if id2label and isinstance(labels[0], str):
    true_labels = np.array([label2id[label] for label in labels])
else:
    # Assume labels are integers from 0 to num_classes-1
    assert all(0 <= label < num_classes for label in true_labels), "Labels must be integers between 0 and num_classes-1"

# Step 5: Tokenize the validation texts
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Step 6: Compute predicted probabilities for each model
all_model_probs = []
for model_dir in model_dirs:
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    all_model_probs.append(probs)

# Step 7: Evaluate ROC AUC using Hugging Face Evaluate (OvO)
roc_auc_metric = load("roc_auc", "multiclass")
for idx, (model_name, probs) in enumerate(zip(model_names, all_model_probs)):
    roc_auc_score = roc_auc_metric.compute(
        references=true_labels,
        prediction_scores=probs,
        multi_class="ovo",
        average="macro"
    )["roc_auc"]
    print(f"Model {model_name} - Macro-average OvO ROC AUC: {roc_auc_score:.4f}")

# Step 8: Generate OvO pairs for plotting
ovo_pairs = list(combinations(range(num_classes), 2))

# Step 9: Plot ROC curves for each OvO pair
num_pairs = len(ovo_pairs)
ncols = min(3, num_pairs)  # Up to 3 columns
nrows = (num_pairs + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), constrained_layout=True)
axes = np.array(axes).flatten() if num_pairs > 1 else [axes]

for ax, (i, j) in zip(axes, ovo_pairs):
    # Select samples belonging to classes i or j
    mask = (true_labels == i) | (true_labels == j)
    selected_labels = true_labels[mask]
    binary_labels = (selected_labels == i).astype(int)  # 1 if class i, 0 if class j
    
    # Compute and plot ROC curve for each model
    for k, model_probs in enumerate(all_model_probs):
        selected_probs = model_probs[mask, i]
        fpr, tpr, _ = roc_curve(binary_labels, selected_probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{model_names[k]} (AUC = {roc_auc:.2f})')
    
    # Add random chance line
    ax.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    class_name_i = id2label[i] if id2label else f'Class {i}'
    class_name_j = id2label[j] if id2label else f'Class {j}'
    ax.set_title(f'{class_name_i} vs {class_name_j}')
    ax.legend(loc='lower right', fontsize='small')

# Hide unused subplots
for ax in axes[num_pairs:]:
    ax.set_visible(False)

plt.show()