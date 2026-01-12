# Import required libraries
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import torch

# Load a text generation model (e.g., causal LM like GPT-based)
model_name = "Qwen/Qwen3-0.6B"  # Ensure your model is suitable for text generation
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRA (Low-Rank Adaptation) Configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Change the task type to Causal Language Modeling
    r=16,           # LoRA rank; typically set to values like 8, 16, or 32
    lora_alpha=32,  # Alpha scaling factor; controls update magnitude
    lora_dropout=0.1,  # Dropout rate for better generalization
    target_modules=['q_proj', 'v_proj'],  # Specific target layers in attention
)

# Apply LoRA configuration
model = get_peft_model(model, lora_config)

# Set special tokens (important for GPT-like models)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Load the IMDB dataset
dataset = load_dataset("imdb")

# Tokenization function for preprocessing
def preprocess_function(examples):
    """
    For causal language modeling, we align inputs and labels. 
    The model should predict the next token based on inputs.
    """
    # Input: Use the review text as input prompt
    inputs = ["Summarize the sentiment of this review: " + review for review in examples["text"]]

    # Tokenization for inputs
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        padding="max_length",
        truncation=True,
    )

    # For causal LM, labels should be the same as the inputs
    # Shift labels to the right inside the model during training
    model_inputs["labels"] = tokenizer(
        inputs,
        max_length=512,
        padding="max_length",
        truncation=True,
    )["input_ids"]

    return model_inputs

# Apply preprocessing to datasets
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["text"])

# Prepare train and evaluation datasets
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# Define data collator to dynamically pad sequences
# Define the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # It's causal language modeling, so we don't use masked LM
)
# Load a generation evaluation metric, for example, ROUGE or BLEU
rouge_metric = evaluate.load("rouge")

# Function to compute metrics during evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE scores
    results = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {key: value.mid.fmeasure for key, value in results.items()}

# Define training arguments (for text generation, adjust params accordingly)
training_args = TrainingArguments(
    output_dir="./results",               # Directory for saving model and checkpoints
    per_device_train_batch_size=4,        # Smaller batch size for longer text inputs
    per_device_eval_batch_size=16,       # Batch size for evaluation
    learning_rate=5e-5,                   # Adjust learning rate for text generation
    weight_decay=0.01,                    # Weight decay for regularization
    num_train_epochs=3,                   # Adjust training epochs
    do_predict=True,                        # Enable text generation during evaluation
    eval_strategy="epoch",          # Evaluate after every epoch
    save_strategy="epoch",                # Save model checkpoints every epoch
    logging_dir="./logs",                 # Directory to save training logs
    logging_steps=100,                    # Log every 100 steps
    report_to="none",  # Avoid logging to external systems like WandB by default
    push_to_hub=False,  # Disable pushing to the Hugging Face hub
)

# Initialize the Trainer object for text generation
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