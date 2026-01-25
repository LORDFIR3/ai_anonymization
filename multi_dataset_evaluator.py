import numpy as np
import pandas as pd
import json
import ast
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report
from collections import Counter
from colorama import Fore, Style, init

init(autoreset=True)

# --- CONFIGURATION ---
MODEL_CHECKPOINT = "dslim/bert-base-NER" 
BATCH_SIZE = 16
SAMPLE_SIZE = 500  # Smaller sample size per dataset for speed validation

class DatasetNormalizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def normalize(self, dataset_name, split="train"):
        print(f"{Fore.CYAN}--- Processing: {dataset_name} ---")
        
        # Load dataset (try/except for split names)
        try:
             # Load a subset for speed
            ds = load_dataset(dataset_name, split=f"{split}[:{SAMPLE_SIZE}]")
        except:
            try:
                ds = load_dataset(dataset_name, split=f"test[:{SAMPLE_SIZE}]")
            except:
                print(f"{Fore.RED}Could not load {dataset_name}")
                return None

        # Dispatcher based on known dataset structures
        if "ai4privacy" in dataset_name or "NAMANDREWLV" in dataset_name:
            return self._normalize_tokenized(ds, "mbert_text_tokens", "mbert_bio_labels")
        
        elif "Isotonic" in dataset_name:
            return self._normalize_tokenized(ds, "tokenised_text", "bio_labels")
        
        elif "nvidia" in dataset_name:
            return self._normalize_spans(ds, text_col="text", spans_col="spans")
            
        elif "gretelai" in dataset_name:
            return self._normalize_spans(ds, text_col="generated_text", spans_col="pii_spans")
        
        else:
            print(f"{Fore.RED}Unknown dataset structure for {dataset_name}")
            return None

    def _normalize_tokenized(self, dataset, tokens_col, labels_col):
        """Simply renames columns to 'tokens' and 'labels'"""
        def clean(example):
            return {
                "tokens": example[tokens_col],
                "labels": example[labels_col]
            }
        return dataset.map(clean, remove_columns=dataset.column_names)

    def _normalize_spans(self, dataset, text_col, spans_col):
        """Converts Text + Character Spans -> Tokens + BIO Labels"""
        print(f"{Fore.YELLOW}Converting Spans to BIO tags for {text_col}...")
        
        def process(example):
            text = example[text_col]
            spans_raw = example[spans_col]
            
            # Parse spans if they are strings
            if isinstance(spans_raw, str):
                try:
                    # Try json first, then ast for single quotes
                    spans = json.loads(spans_raw)
                except:
                    try:
                        spans = ast.literal_eval(spans_raw)
                    except:
                        spans = [] # Fail safe
            else:
                spans = spans_raw # Already list/dict
            
            # Tokenize text while keeping track of offsets
            tokenized = self.tokenizer(text, return_offsets_mapping=True, truncation=True)
            tokens = self.tokenizer.convert_ids_to_tokens(tokenized["input_ids"])
            offsets = tokenized["offset_mapping"]
            
            # Initialize labels as 'O'
            labels = ["O"] * len(tokens)
            
            # Ensure spans is a list of dicts: [{'start': 0, 'end': 10, 'label': 'PER'}]
            # Some datasets might have different span structures, let's normalize this list
            clean_spans = []
            if isinstance(spans, list):
                for s in spans:
                    # Nvidia style: {'start_position', 'end_position', 'label'}
                    # Gretel style: {'start', 'end', 'label'} or list of dicts
                    start = s.get('start', s.get('start_position'))
                    end = s.get('end', s.get('end_position'))
                    label = s.get('label', s.get('entity_type'))
                    
                    if start is not None and end is not None and label:
                         clean_spans.append((int(start), int(end), label))
            
            # Map Spans to Tokens
            for (start_char, end_char, label_str) in clean_spans:
                for idx, (token_start, token_end) in enumerate(offsets):
                    # Check overlap
                    if token_start == 0 and token_end == 0: continue # Skip special tokens
                    
                    if token_start >= start_char and token_end <= end_char:
                        # Logic for B- vs I- tag:
                        # If this token is the *first* one encountered for this span, mark B-
                        # However, iterating linearly and overwriting O is safer
                        
                        # Simplified Check: Is it the beginning of the overlapping region?
                        # Note: This is an approximation. A robust aligner is complex. 
                        # We will use: if current tag is O, make it B-, else I- (if contiguous)
                        # Actually simpler: standard B/I logic based on start char
                        
                        if token_start == start_char or (token_start > start_char and labels[idx-1] == "O"):
                             # Attempt to mark B- if exact match or start
                             labels[idx] = f"B-{label_str}"
                        else:
                             # If previous was B- or I- of same label, mask I-
                             # But here we just broadly assume intersection = entity
                             if labels[idx] == "O": # Don't overwrite existing
                                 if idx > 0 and (labels[idx-1].startswith(f"B-{label_str}") or labels[idx-1].startswith(f"I-{label_str}")):
                                     labels[idx] = f"I-{label_str}"
                                 else:
                                     labels[idx] = f"B-{label_str}"

            return {
                "tokens": tokens,
                "labels": labels
            }

        return dataset.map(process, remove_columns=dataset.column_names)


class UniversalEvaluator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        self.model = AutoModelForTokenClassification.from_pretrained(MODEL_CHECKPOINT)
        self.normalizer = DatasetNormalizer(self.tokenizer)
        
        self.model_label2id = self.model.config.label2id
        self.model_id2label = self.model.config.id2label

    def map_pii_to_ner_schema(self, label):
        """Maps PII -> Generic NER (PER, LOC, ORG)"""
        # Normalize casing
        label_upper = label.upper()
        
        if label_upper.startswith("B-") or label_upper.startswith("I-"):
            prefix, name = label_upper[:2], label_upper[2:]
            
            # Person
            if any(x in name for x in ["NAME", "USER", "ACCOUNT"]):
                return f"{prefix}PER"
            # Location
            if any(x in name for x in ["CITY", "STATE", "COUNTRY", "ADDRESS", "LOC", "GPS", "ZIP"]):
                return f"{prefix}LOC"
            # Org
            if any(x in name for x in ["COMPANY", "ORG", "ISSUER", "BANK"]):
                return f"{prefix}ORG"
                
        return label # Fallback

    def evaluate(self, dataset_name):
        ds = self.normalizer.normalize(dataset_name)
        if ds is None: return

        print(f"{Fore.GREEN}✔ Normalized {len(ds)} samples.")
        
        # Helper to align for BERT (Tokenizing again? No, we already have tokens)
        # We need to turn 'tokens' list into 'input_ids' for the model.
        # Since we might have re-tokenized in _normalize_spans, tokens are model-ready pieces?
        # WAIT: _normalize_tokenized gives WORDS (pre-tokenized space split usually), NOT subwords.
        # _normalize_spans gives SUBWORDS (using self.tokenizer).
        
        # To be safe for both: We treat them as "words" and use is_split_into_words=True.
        # For _normalize_spans, they are already subwords, but passing them as words is fine (sub-sub-tokenizing is idempotent mostly).
        
        def tokenize_and_align(examples):
            tokenized_inputs = self.tokenizer(
                examples["tokens"], 
                truncation=True, 
                is_split_into_words=True
            )
            
            # Align labels
            labels = []
            ref_labels = []
            for i, label_list in enumerate(examples["labels"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                ref_ids = []
                
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                        ref_ids.append("PAD")
                    elif word_idx != previous_word_idx:
                        # New word start
                        # Handle potential length mismatch if dataset tokens != tokenizer tokens
                        # But label_list is aligned to examples["tokens"]
                        if word_idx < len(label_list):
                            tag = label_list[word_idx]
                        else:
                            tag = "O"
                            
                        ref_ids.append(tag)
                        
                        # Map to Model ID
                        generic_tag = self.map_pii_to_ner_schema(tag) # Try to map first?
                        # Actually model expects exact match B-PER. 
                        # My map_pii_to_ner_schema gives B-PER.
                        # So let's check if mapped tag is in model
                        if generic_tag in self.model_label2id:
                             label_ids.append(self.model_label2id[generic_tag])
                        else:
                             label_ids.append(0) # O
                    else:
                        label_ids.append(-100) # Subword
                        ref_ids.append("SUBWORD")
                    previous_word_idx = word_idx
                labels.append(label_ids)
                ref_labels.append(ref_ids)
            
            tokenized_inputs["labels"] = labels
            tokenized_inputs["reference_labels"] = ref_labels
            return tokenized_inputs

        tokenized_ds = ds.map(tokenize_and_align, batched=True)
        
        # Predict
        args = TrainingArguments(output_dir="./results_tmp", report_to="none", per_device_eval_batch_size=BATCH_SIZE)
        trainer = Trainer(model=self.model, args=args, tokenizer=self.tokenizer, data_collator=DataCollatorForTokenClassification(self.tokenizer))
        
        predictions, _, _ = trainer.predict(tokenized_ds)
        preds = np.argmax(predictions, axis=2)
        
        # Extract Flat Lists
        true_labels_flat = []
        pred_labels_flat = []
        
        # Get Reference Labels for ground truth
        ref_all = tokenized_ds["reference_labels"]
        
        for i in range(len(ref_all)):
            for j in range(len(ref_all[i])):
                lbl = ref_all[i][j]
                if lbl not in ["PAD", "SUBWORD"]:
                    true_labels_flat.append(lbl) # Detailed PII tag
                    
                    # Pred
                    p_id = preds[i][j]
                    pred_labels_flat.append(self.model_id2label.get(p_id, "O")) # Generic NER tag

        # Generate Reports
        print(f"\n{Fore.YELLOW}Results for {dataset_name}:")
        
        # Mapped Report
        mapped_true = [self.map_pii_to_ner_schema(l) for l in true_labels_flat]
        labels_in_pred = sorted(list(set(mapped_true + pred_labels_flat)))
        # Filter for only generic classes
        labels_generic = [l for l in labels_in_pred if "PER" in l or "LOC" in l or "ORG" in l or "O" in l]
        
        print(classification_report(mapped_true, pred_labels_flat, labels=labels_generic, zero_division=0))
        print("-" * 40)

# --- EXECUTION ---
datasets_to_check = [
    "ai4privacy/pii-masking-200k",
    "nvidia/Nemotron-PII",
    "Isotonic/pii-masking-200k",
    # "NAMANDREWLV/pii-masking-200k", # Likely same as ai4privacy, skipping to save time
    "gretelai/synthetic_pii_finance_multilingual"
]

evaluator = UniversalEvaluator()
for ds in datasets_to_check:
    evaluator.evaluate(ds)
