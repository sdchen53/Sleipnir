# Import required libraries
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import numpy as np
import torch

# Load the model for sequence classification
model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRA (Low-Rank Adaptation) Configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,  # LoRA rank; typically set to values like 8, 16, or 32
    lora_alpha=32,  # Alpha scaling factor; controls update magnitude
    lora_dropout=0.1,  # Dropout rate for better generalization
    target_modules=['q_proj', 'v_proj'],  # Target specific layers (e.g., query/value for attention)
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Load a dataset (for example, the Glue SST-2 dataset for sequence classification)
dataset = load_dataset("glue", "sst2")

# Tokenization function for preprocessing
def preprocess_function(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=128,  # Limit input sequence length
        padding="max_length",
    )

# Preprocess the data
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Prepare train and evaluation datasets
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

# Data collator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load a metric for evaluation (e.g., accuracy)
accuracy_metric = evaluate.load("accuracy")

# Function to compute metrics during evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save the model and checkpoints
    eval_strategy="epoch",  # Evaluate at the end of every epoch
    learning_rate=2e-4,  # LoRA benefits from slightly higher learning rates
    per_device_train_batch_size=32,  # Batch size for training
    per_device_eval_batch_size=64,  # Batch size for evaluation
    num_train_epochs=3,  # Fine-tune for a few epochs
    weight_decay=0.01,  # Weight decay for regularization
    logging_dir="./logs",  # Directory to save logs
    logging_steps=100,  # Log every 100 steps
    save_strategy="epoch",  # Save model at the end of each epoch
    save_total_limit=2,  # Limit the number of saved checkpoints
    report_to="none",  # Avoid logging to external systems like WandB by default
    push_to_hub=False,  # Disable pushing to the Hugging Face hub
)

# Initialize the Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the LoRA fine-tuned model
model.save_pretrained("./lora_finetuned_model")
tokenizer.save_pretrained("./lora_finetuned_model")