#%%
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback
)
from datasets import Dataset
import json
import os

# 4. Define a custom callback to save metrics
class SaveMetricsCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.metrics_history = {"train": [], "eval": []}
        os.makedirs(output_dir, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Called whenever logs are generated (e.g., loss)
        if logs is not None:
            epoch = state.epoch
            step = state.global_step
            log_entry = {"epoch": epoch, "step": step, **logs}
            if "loss" in logs:  # Training logs
                self.metrics_history["train"].append(log_entry)
            elif "eval_loss" in logs:  # Evaluation logs
                self.metrics_history["eval"].append(log_entry)
            elif "accuracy" in logs:
                self.metrics_history["accuracy"].append(log_entry)

    def on_train_end(self, args, state, control, **kwargs):
        # Save metrics to JSON files at the end of training
        with open(os.path.join(self.output_dir, "train_metrics.json"), "w") as f:
            json.dump(self.metrics_history["train"], f, indent=4)
        with open(os.path.join(self.output_dir, "eval_metrics.json"), "w") as f:
            json.dump(self.metrics_history["eval"], f, indent=4)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(-1)  # Get predicted class
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

#%%
def prepare_dataset(
    csv_path, 
    text_column='text', 
    label_column='label', 
    model_name="intfloat/e5-small-v2"
):
    """
    Comprehensive dataset preparation with modern Hugging Face practices
    """
    # 1. Load CSV
    df = pd.read_csv(csv_path).drop(columns=['class_name']).dropna().drop_duplicates()
    df['text'] = df['text'].apply(lambda x: x.replace('\n', ''))
    # Add index column
    df['input_ids'] = range(len(df))
    df = df.iloc[:, [2, 0, 1]] 

    # Validate columns
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found. Available columns: {df.columns.tolist()}")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Available columns: {df.columns.tolist()}")
    
    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 3. Preprocessing function
    def preprocess_function(examples):
        # Tokenize texts
        tokenized = tokenizer(
            examples[text_column], 
            truncation=True, 
            padding=False  # Let DataCollator handle padding
        )
        
        # Add labels
        tokenized['labels'] = examples[label_column]
        
        return tokenized
    
    # 4. Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    # 5. Tokenize dataset
    tokenized_dataset = dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset.column_names
    )
    
    # 6. Split dataset
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
    
    # 7. Prepare data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, 
        padding=True
    )
    
    # 8. Determine number of labels
    num_labels = len(set(df[label_column]))
    
    return {
        'train_dataset': split_dataset['train'],
        'eval_dataset': split_dataset['test'],
        'tokenizer': tokenizer,
        'data_collator': data_collator,
        'num_labels': num_labels
    }

#%%

def fine_tune_model(
    csv_path, 
    layer_freeze,
    layer_not_freeze,
    text_column='text', 
    label_column='label', 
    model_name="intfloat/e5-small-v2",
    output_dir="./fine_tuned_model"
):
    """
    Modern fine-tuning approach with updated Hugging Face best practices
    """
    # Prepare dataset
    dataset_prep = prepare_dataset(
        csv_path, 
        text_column, 
        label_column, 
        model_name
    )
    
    # Load model with precise number of labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=dataset_prep['num_labels']
    )
#     for name, param in model.named_parameters():
#         param.requires_grad = False 
# # Stages 1-N: unfreeze encoder layers progressively
#     for layer_idx in range(min(layer_freeze, 12)):
#         for param in model.encoder.layer[layer_idx].parameters():
#             param.requires_grad = True
#         print(f"Encoder layer {layer_idx} unfrozen")
    for name, param in model.named_parameters():
        if layer_freeze not in name: #!! For testing only embedding layer and attention
            param.requires_grad = False
        if layer_not_freeze in name:
            param.requires_grad = True
            # print(f"{name}: {'trainable' if param.requires_grad else 'frozen'}")
    for name, param in model.named_parameters():
        print(f"{name}: {'trainable' if param.requires_grad else 'frozen'}")

    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Initialize Trainer with modern approach
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_prep['train_dataset'],
        eval_dataset=dataset_prep['eval_dataset'],
        tokenizer=dataset_prep['tokenizer'],
        data_collator=dataset_prep['data_collator'],
        compute_metrics=compute_metrics,
        callbacks=[SaveMetricsCallback(output_dir)],
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(output_dir)
    
    # Evaluate
    # 9. Save final results manually (optional)
    final_results = trainer.evaluate()
    print(f"Evaluation Results: {final_results}")

    with open(os.path.join(output_dir, "final_eval_results.json"), "w") as f:
        json.dump(final_results, f, indent=4)
    
    return trainer

#%%
# Example usage
if __name__ == "__main__":
    # for i in range(12):
    try:
        fine_tune_model(
            csv_path='merged_test_train.csv',  # Replace with your CSV path
            layer_freeze='None',  # Replace with your desired layer to freeze
            layer_not_freeze='encoder.layer.11',  # Replace with your desired layer to not freeze
            text_column='text',  # Replace with your text column name
            label_column='label',  # Replace with your label column name
            output_dir='./fine_tuned_model_attention11',  # Replace with your desired output directory
            model_name='BAAI/bge-small-en-v1.5'
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

print("Modern Hugging Face Fine-Tuning Script Ready!")