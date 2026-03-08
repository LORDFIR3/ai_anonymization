import numpy as np
import evaluate
import ast
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)

# ==========================================
# 1. Configuration
# ==========================================
# Make sure to use the exact Hugging Face identifiers
model_id = "OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1" # Replace with exact HF path if different
dataset_id = "nvidia/Nemotron-PII"

# ==========================================
# 2. Load Dataset & Tokenizer
# ==========================================
dataset = load_dataset(dataset_id)
tokenizer = AutoTokenizer.from_pretrained(model_id, add_prefix_space=True)

# Extract unique labels from the dataset
unique_labels = set()
print("Scanning dataset for unique labels...")
# Scan a subset of train dataset to get all labels
for example in dataset['train'].select(range(min(5000, len(dataset['train'])))):
    spans = ast.literal_eval(example["spans"]) if isinstance(example["spans"], str) else example["spans"]
    for span in spans:
        unique_labels.add(span["label"])

unique_labels = sorted(list(unique_labels))
label_list = ["O"]
for label in unique_labels:
    label_list.append(f"B-{label}")
    label_list.append(f"I-{label}")

num_labels = len(label_list)

# Setup label dictionaries for the model
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

# ==========================================
# 3. Preprocessing (Tokenization & Label Alignment)
# ==========================================
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        return_offsets_mapping=True, 
        padding="max_length"
    )

    labels = []
    for i, (offsets, spans_str) in enumerate(zip(tokenized_inputs["offset_mapping"], examples["spans"])):
        label_ids = [0] * len(offsets) # Default all to 'O'
        sequence_ids = tokenized_inputs.sequence_ids(i)
        
        for idx, (offset, seq_id) in enumerate(zip(offsets, sequence_ids)):
            if seq_id is None:
                label_ids[idx] = -100

        spans = ast.literal_eval(spans_str) if isinstance(spans_str, str) else spans_str
        sorted_spans = sorted(spans, key=lambda x: x["start"])
        
        for span in sorted_spans:
            start_char = span["start"]
            end_char = span["end"]
            label_str = span["label"]
            
            # safeguard in case label wasn't in the scan
            if f"B-{label_str}" not in label2id:
                continue
            
            b_label_id = label2id[f"B-{label_str}"]
            i_label_id = label2id[f"I-{label_str}"]
            
            span_started = False
            for idx, (offset_start, offset_end) in enumerate(offsets):
                if sequence_ids[idx] is None or offset_start == offset_end:
                    continue
                
                if offset_start >= start_char and offset_end <= end_char:
                    if not span_started:
                        label_ids[idx] = b_label_id
                        span_started = True
                    else:
                        label_ids[idx] = i_label_id
                elif offset_start < end_char and offset_end > start_char:
                     if not span_started:
                        label_ids[idx] = b_label_id
                        span_started = True
                     else:
                        label_ids[idx] = i_label_id

        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs

# Apply preprocessing dynamically
print("Tokenizing datasets...")
# Using a subset for faster demonstration and testing
train_dataset = dataset["train"].select(range(1000))
eval_dataset = dataset["test"].select(range(200))
tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval = eval_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=eval_dataset.column_names)

# ==========================================
# 4. Load Model
# ==========================================
# ignore_mismatched_sizes=True is crucial! It throws away the OpenMed model's 
# old classification head and creates a new one fitted perfectly to Nemotron's labels.
model = AutoModelForTokenClassification.from_pretrained(
    model_id, 
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True 
)

# ==========================================
# 5. Training Setup
# ==========================================
training_args = TrainingArguments(
    output_dir="./nemotron-finetuned-model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True, # Set to True if you have a GPU
)

# Data collator automatically pads inputs and labels to the max length in the batch
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Use seqeval for standard NER metrics (Precision, Recall, F1)
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (-100)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval, # Assuming dataset has 'test' split
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ==========================================
# 6. Run Training & Save
# ==========================================
trainer.train()
trainer.save_model("./nemotron-finetuned-model-final")
tokenizer.save_pretrained("./nemotron-finetuned-model-final")
