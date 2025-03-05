#%%
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset
import datasets
import pandas as pd

# Load dataset
# dataset = load_dataset("imdb")

# Load tokenizer and model
model_name = "intfloat/e5-small-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

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

tok_oup=dataset.map(lambda x:tokenizer(x['text'],  padding='max_length'), batched=True)
#%%
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
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation']
)


#%%
# Train the model
trainer.train()


# Save the fine-tuned model
trainer.save_model("./craigslist_bargains_fine_tuned_full")

# Optional: Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")