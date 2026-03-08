import ast
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import classification_report

# Configuration
MODEL_PATH = "./ai4privacy-deberta-nemotron/checkpoint-7500"
DATASET_ID = "nvidia/Nemotron-PII"
MAX_SEQ_LENGTH = 128
NUM_SAMPLES = 2000 # Evaluate on the whole 20% by default

def create_tokenize_fn(tokenizer, label2id):
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_offsets_mapping=True,
        )
        
        labels_batch = []
        for i, (offsets, spans_str) in enumerate(zip(tokenized_inputs["offset_mapping"], examples["spans"])):
            label_ids = [0] * len(offsets)
            sequence_ids = tokenized_inputs.sequence_ids(i)
            
            # Mask special tokens
            for idx, seq_id in enumerate(sequence_ids):
                if seq_id is None:
                    label_ids[idx] = -100
                    
            spans = ast.literal_eval(spans_str) if isinstance(spans_str, str) else spans_str
            sorted_spans = sorted(spans, key=lambda x: x["start"])
            
            for span in sorted_spans:
                b_label = f"B-{span['label']}"
                i_label = f"I-{span['label']}"
                
                if b_label not in label2id:
                    continue
                    
                span_started = False
                for idx, (offset_start, offset_end) in enumerate(offsets):
                    if label_ids[idx] == -100 or offset_start == offset_end:
                        continue
                        
                    if offset_start < span["end"] and offset_end > span["start"]:
                        if not span_started:
                            label_ids[idx] = label2id[b_label]
                            span_started = True
                        else:
                            label_ids[idx] = label2id[i_label]
                            
            labels_batch.append(label_ids)
            
        tokenized_inputs["labels"] = labels_batch
        tokenized_inputs.pop("offset_mapping", None)
        return tokenized_inputs
        
    return tokenize_and_align_labels

def main():
    print(f"Loading local model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    
    id2label = model.config.id2label
    label2id = model.config.label2id
    
    print(f"Loading dataset {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID, split="test")
    
    # Same 20% slice used in train_deberta.py
    test_size = int(len(dataset) * 0.2)
    dataset = dataset.select(range(test_size))
    
    if NUM_SAMPLES:
        dataset = dataset.select(range(min(NUM_SAMPLES, len(dataset))))
        print(f"Using a subset of {NUM_SAMPLES} samples for quick evaluation.")
    else:
        print(f"Evaluating on the full 20% test slice ({len(dataset)} samples).")
    
    print("Tokenizing dataset...")
    tokenize_fn = create_tokenize_fn(tokenizer, label2id)
    tokenized_eval = dataset.map(
        tokenize_fn, batched=True, num_proc=4, remove_columns=dataset.column_names, desc="Tokenizing Test Set"
    )
    
    # Configure Trainer purely for prediction
    args = TrainingArguments(
        output_dir="./results_tmp", 
        per_device_eval_batch_size=16, 
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer)
    )
    
    print("Running batch predictions (this may take a few minutes)...")
    predictions, labels, _ = trainer.predict(tokenized_eval)
    preds = np.argmax(predictions, axis=2)
    
    print("Flattening labels for classification report...")
    true_labels_flat = []
    pred_labels_flat = []
    
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] != -100:  # Ignore special tokens and padding
                true_tag = id2label[labels[i][j]]
                pred_tag = id2label[preds[i][j]]
                true_labels_flat.append(true_tag)
                pred_labels_flat.append(pred_tag)
                
    # Strip B- and I- prefixes to group by the underlying entity class
    def strip_bio_prefix(tag):
        if tag.startswith("B-") or tag.startswith("I-"):
            return tag[2:]
        return tag

    true_report = [strip_bio_prefix(t) for t in true_labels_flat]
    pred_report = [strip_bio_prefix(t) for t in pred_labels_flat]
    
    # Filter out 'O' to only show PII entities in the report
    labels_in_data = sorted(list(set(true_report + pred_report)))
    target_names = [l for l in labels_in_data if l not in ["O", "PAD"]]
    
    print("\n" + "="*80)
    print(f"Classification Report: {MODEL_PATH} on 20% of {DATASET_ID}")
    print("="*80)
    
    if not target_names:
        print("No entities found!")
    else:
        print(classification_report(true_report, pred_report, labels=target_names, zero_division=0))

if __name__ == "__main__":
    main()
