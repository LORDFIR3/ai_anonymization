import ast
import numpy as np
import evaluate
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)

# Configuration Constants
MODEL_ID = "microsoft/deberta-v3-small"
DATASET_ID = "nvidia/Nemotron-PII"
MAX_SEQ_LENGTH = 256
OUTPUT_DIR = "./ai4privacy-deberta-nemotron"

def get_unique_labels(dataset, num_samples=5000):
    """Extracts unique bio tags from a subset of the dataset."""
    print("Extracting unique labels...")
    unique_labels = set()
    
    # Scan a sample of the train dataset
    subset_size = min(num_samples, len(dataset['train']))
    subset = dataset['train'].select(range(subset_size))
    for example in tqdm(subset, desc="Extracting unique labels", total=subset_size):
        spans = ast.literal_eval(example["spans"]) if isinstance(example["spans"], str) else example["spans"]
        for span in spans:
            unique_labels.add(span["label"])
            
    # Build B- and I- prefixes
    label_list = ["O"]
    for label in sorted(list(unique_labels)):
        label_list.extend([f"B-{label}", f"I-{label}"])
        
    print(f"Found {len(label_list)} total BIO labels.")
    return label_list

def create_tokenize_fn(tokenizer, label2id):
    """Returns a function to tokenize inputs and align BIO labels."""
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_offsets_mapping=True,
            # Removed padding="max_length" to allow dynamic padding via DataCollator, improving training speed.
        )
        
        labels_batch = []
        for i, (offsets, spans_str) in enumerate(zip(tokenized_inputs["offset_mapping"], examples["spans"])):
            label_ids = [0] * len(offsets)
            sequence_ids = tokenized_inputs.sequence_ids(i)
            
            # Mask special tokens with -100
            for idx, seq_id in enumerate(sequence_ids):
                if seq_id is None:
                    label_ids[idx] = -100
                    
            # Parse spans
            spans = ast.literal_eval(spans_str) if isinstance(spans_str, str) else spans_str
            sorted_spans = sorted(spans, key=lambda x: x["start"])
            
            # Map character spans to token labels
            for span in sorted_spans:
                b_label = f"B-{span['label']}"
                i_label = f"I-{span['label']}"
                
                if b_label not in label2id:
                    continue
                    
                span_started = False
                for idx, (offset_start, offset_end) in enumerate(offsets):
                    if label_ids[idx] == -100 or offset_start == offset_end:
                        continue
                        
                    # Target span vs token offsets
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

def get_compute_metrics_fn(id2label):
    """Returns a sequence evaluation metric computation function."""
    seqeval = evaluate.load("seqeval")
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens) and map to string labels
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        
    return compute_metrics

def main():
    print(f"[1/5] Loading dataset {DATASET_ID}...")
    print("      (This may take a few minutes to download and cache if it's your first time doing so)")
    dataset = load_dataset(DATASET_ID)
    
    print("\n      (Subsetting to 20% of the dataset for faster training)")
    train_size = int(len(dataset["train"]) * 0.2)
    test_size = int(len(dataset["test"]) * 0.2)
    dataset["train"] = dataset["train"].select(range(train_size))
    dataset["test"] = dataset["test"].select(range(test_size))
    
    print(f"\n[2/5] Loading tokenizer {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    
    # Setup labels mappings
    label_list = get_unique_labels(dataset)
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    
    # Tokenize datasets
    print("\n[3/5] Tokenizing datasets... (Progress bars below)")
    tokenize_fn = create_tokenize_fn(tokenizer, label2id)
    
    tokenized_train = dataset["train"].map(
        tokenize_fn, batched=True, num_proc=4, remove_columns=dataset["train"].column_names, desc="Tokenizing Train"
    )
    tokenized_eval = dataset["test"].map(
        tokenize_fn, batched=True, num_proc=4, remove_columns=dataset["test"].column_names, desc="Tokenizing Test"
    )

    # Load Model
    print(f"\n[4/5] Loading model {MODEL_ID}...")
    print("      (This is a very large file, downloading weights may take some time. A progress bar will appear.)")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    # -------------------------------------------------------------
    # GPU MEMORY OPTIMIZATIONS for 4GB / Low VRAM
    # -------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        
        # 1. Physical vs Effective Batch Size
        per_device_train_batch_size=1,     # VRAM Opt: Reduce physical batch to minimum
        gradient_accumulation_steps=8,     # VRAM Opt: Keeps effective batch size at 8 (1 * 8)
        per_device_eval_batch_size=8,      # Speed Opt: Eval has no gradients, can use a larger batch size
        
        # 1.5. Dataloader Bottleneck Optimizations
        dataloader_num_workers=4,          # Speed Opt: Multiprocessing data loading prevents GPU starvation
        dataloader_pin_memory=True,        # Speed Opt: Faster CPU to GPU data transfer
        
        # 2. Gradient Checkpointing
       gradient_checkpointing=True,       # VRAM Opt: Trades a little compute time for massive memory saving
        gradient_checkpointing_kwargs={"use_reentrant": False}, # Fixes the "Trying to backward through the graph a second time" error
        
        # 3. 8-bit Optimizer
        optim="adamw_8bit",                # VRAM Opt: Saves optimizer state memory (requires pip install bitsandbytes)
        
        # 4. FP16 Precision
        fp16=True,                         # VRAM/Speed Opt: DeBERTa runs much faster in mixed precision
        
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=100
    )
    
    # Setup Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=get_compute_metrics_fn(id2label),
    )
    
    print("\n[5/5] Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model(f"{OUTPUT_DIR}-final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}-final")
    print(f"Model successfully saved to {OUTPUT_DIR}-final")

if __name__ == "__main__":
    main()
