#%%
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset
import datasets
import pandas as pd

def custom_prepare_features(tokenizer, max_length=512):
    """
    Create a robust feature preparation function
    """
    def prepare_features(examples):
        # Ensure texts are strings
        texts = [str(text).strip() for text in examples['text']]
        
        # Tokenize with padding and truncation
        encodings = tokenizer(
            texts, 
            padding='max_length', 
            truncation=True, 
            max_length=max_length,
            return_tensors='pt'  # Explicitly return PyTorch tensors
        )
        
        # Ensure all expected keys are present
        features = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
        
        # Add label if present
        if 'labels' in examples:
            features['labels'] = torch.tensor(examples['labels'], dtype=torch.long)
        
        return features
    
    return prepare_features

# Load dataset
# dataset = load_dataset("imdb")
def detailed_tensor_debug(csv_path, text_column='text', label_column='label', model_name="intfloat/e5-small-v2"):
    """
    Comprehensive tensor debugging script
    """
    # 1. Load the CSV
    df = pd.read_csv(csv_path)
    
    # Print initial dataframe info
    print("\n--- Dataset Information ---")
    print(f"Total rows: {len(df)}")
    print("Columns:", df.columns.tolist())
    
    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 3. Tokenization debugging
    print("\n--- Tokenization Debugging ---")
    
    # Take first few rows for testing
    sample_texts = df[text_column].head().tolist()
    
    # Perform manual tokenization
    try:
        # Basic tokenization
        print("Basic Tokenization:")
        encoded = tokenizer(sample_texts, padding=True, truncation=True, return_tensors='pt')
        
        # Print details of encoded tensors
        print("\nEncoded Tensor Details:")
        for key, tensor in encoded.items():
            print(f"{key} shape: {tensor.shape}")
            print(f"{key} type: {type(tensor)}")
        
        # Specific input_ids investigation
        input_ids = encoded['input_ids']
        print("\nInput IDs specifics:")
        print(f"Number of dimensions: {input_ids.ndim}")
        print(f"Shape: {input_ids.shape}")
        
    except Exception as e:
        print(f"Tokenization Error: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Manual tensor shape investigation
    print("\n--- Manual Tensor Investigation ---")
    def manual_tensor_check(texts):
        """
        Manually create and check tensors
        """
        try:
            # Tokenize texts
            encodings = tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            )
            
            # Detailed tensor information
            for key, tensor in encodings.items():
                print(f"\n{key} Tensor:")
                print(f"  Shape: {tensor.shape}")
                print(f"  Dimensions: {tensor.ndim}")
                print(f"  Data type: {tensor.dtype}")
            
            return encodings
        
        except Exception as e:
            print(f"Manual tensor check error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Perform manual check
    manual_tensor_check(sample_texts)
    
    # 5. Potential workaround
    print("\n--- Potential Workaround Suggestions ---")
    print("1. Ensure consistent text length")
    print("2. Use max_length parameter in tokenizer")
    print("3. Check for any irregular data in text column")
    
    # Bonus: Check for any problematic texts
    print("\n--- Potential Problematic Texts ---")
    problematic_texts = df[df[text_column].apply(lambda x: len(str(x)) == 0)]
    if len(problematic_texts) > 0:
        print("Found empty texts:")
        print(problematic_texts)
    else:
        print("No empty texts found.")

# Load tokenizer and model
model_name = "intfloat/e5-small-v2"
# Prepare feature extraction function
tokenizer = AutoTokenizer.from_pretrained(model_name)

    
    # Load model with explicit number of labels


#%%
from datasets import load_dataset

dataset = DatasetDict()

df = pd.read_csv("data/train.csv").drop(columns=['class_name']).dropna().drop_duplicates()
df['text'] = df['text'].apply(lambda x: x.replace('\n', ''))
df = df.rename(columns={'label': 'labels'})
# Add index column
df['input_ids'] = range(len(df))
df = df.iloc[:, [2, 0, 1]]  # 'Sr.no', 'Maths Score', 'Name'
dataset['train'] = Dataset.from_pandas(df)
unique_labels = len(set(df['labels'].unique()))

df = pd.read_csv("data/test.csv").drop(columns=['class_name']).dropna().drop_duplicates()
df['text'] = df['text'].apply(lambda x: x.replace('\n', ''))
df = df.rename(columns={'label': 'labels'})
# Add index column
df['input_ids'] = range(len(df))
df = df.iloc[:, [2, 0, 1]]  # 'Sr.no', 'Maths Score', 'Name'

dataset['test'] = Dataset.from_pandas(df)

df = pd.read_csv("data/test.csv").drop(columns=['class_name']).dropna().drop_duplicates()
df['text'] = df['text'].apply(lambda x: x.replace('\n', ''))
df = df.rename(columns={'label': 'labels'})
df['input_ids'] = range(len(df))
df = df.iloc[:, [2, 0, 1]]  # 'Sr.no', 'Maths Score', 'Name'
# Add index column
dataset['validation'] = Dataset.from_pandas(df)
#%%    # Map features with comprehensive error handling
# Prepare feature extraction function
prepare_features_fn = custom_prepare_features(tokenizer)
    

try:
    tokenized_datasets = dataset.map(
        prepare_features_fn, 
        batched=True, 
        remove_columns=dataset['train'].column_names
    )
except Exception as e:
    print(f"Tokenization Error: {e}")
    import traceback
    traceback.print_exc()
    raise
# tok_oup=dataset.map(lambda x:tokenizer(x['text'],  padding='max_length'), batched=True)
#%%

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=unique_labels
)

for idx, (name, param) in enumerate(model.named_parameters()):
    # if idx < 10:
    print(name)
# %%
# Training arguments
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

training_args = TrainingArguments(
    output_dir="./craigslist_bargains_results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,
    torch_compile=True,
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit",
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)


#%%
# Train the model
trainer.train()


# Save the fine-tuned model
trainer.save_model("./craigslist_bargains_fine_tuned_full")

# Optional: Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")