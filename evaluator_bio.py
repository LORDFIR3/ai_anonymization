import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report
from colorama import Fore, init, Style
import torch
import random
import sys

init(autoreset=True)

# --- CONFIGURATION ---
BATCH_SIZE = 4
MAX_SAMPLES = 1000 
MIN_PER_CLASS = 10 
SEED = 42

BIO_MODELS = [
    "Isotonic/distilbert_finetuned_ai4privacy_v2",       
    "Isotonic/deberta-v3-base_finetuned_ai4privacy_v2", 
]

BIO_DATASETS = [
    "ai4privacy/pii-masking-200k",
    "Isotonic/pii-masking-200k"
]

class BioEvaluator:
    @staticmethod
    def map_pii_to_ner_schema(label):
        """Maps PII -> Generic NER (PER, LOC, ORG)"""
        label_raw = str(label)
        label_upper = label_raw.upper()
        
        if label_upper.startswith("B-") or label_upper.startswith("I-"):
            prefix = label_upper[:2]
            name = label_upper[2:]
        else:
            prefix = ""
            name = label_upper

        if any(x == name for x in ["ORG", "COMPANY", "COMPANYNAME", "BANK", "ISSUER", "UNIVERSITY"]):
            return f"{prefix}ORG"
        if "COMPANY" in name or "ORG" in name: 
            return f"{prefix}ORG"

        if any(x in name for x in ["PER", "FIRSTNAME", "LASTNAME", "MIDDLENAME", "USERNAME", "NAME", "USER", "ACCOUNT"]):
            if "NUMBER" in name: return label_raw 
            return f"{prefix}PER"

        if any(x in name for x in ["LOC", "CITY", "STATE", "COUNTRY", "ADDRESS", "ZIP", "GPS", "STREET", "BUILDING", "COUNTY"]):
            return f"{prefix}LOC"
            
        return label_raw 

class DatasetNormalizerBIO:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def normalize(self, dataset_name, split="train"):
        print(f"{Fore.CYAN}--- Processing BIO Dataset: {dataset_name} ---")
        
        try:
            ds = load_dataset(dataset_name, split=split)
        except Exception as e:
            print(f"{Fore.RED}Could not load {dataset_name}: {e}")
            return None

        normalized_ds = None
        if "ai4privacy" in dataset_name:
             normalized_ds = self._standardize_columns(ds, "mbert_text_tokens", "mbert_bio_labels")
        elif "Isotonic" in dataset_name:
             normalized_ds = self._standardize_columns(ds, "tokenised_text", "bio_labels")
        else:
             print(f"{Fore.RED}Unknown structure for {dataset_name}")
             return None
             
        print(f"{Fore.YELLOW}Performing stratified sampling (Target: {MIN_PER_CLASS} per class)...")
        sampled_ds = self._stratified_sample(normalized_ds)
        print(f"{Fore.GREEN}Selected {len(sampled_ds)} samples.")
        
        return sampled_ds

    def _standardize_columns(self, dataset, tokens_col, labels_col):
        if tokens_col not in dataset.column_names or labels_col not in dataset.column_names:
             print(f"{Fore.RED}Columns {tokens_col}/{labels_col} not found.")
             return None

        def clean(example):
            return {
                "tokens": example[tokens_col],
                "labels": example[labels_col]
            }
        return dataset.map(clean, remove_columns=dataset.column_names)

    def _stratified_sample(self, dataset):
        shuffled = dataset.shuffle(seed=SEED)
        selected_indices = []
        counts = {"PER": 0, "LOC": 0, "ORG": 0}
        
        print("Sampling iteration...", file=sys.stderr)
        
        for i, example in enumerate(shuffled):
            if len(selected_indices) >= MAX_SAMPLES:
                break
            
            if i > 0 and i % 10000 == 0:
                print(f"Scanned {i} examples... Found: {counts}", file=sys.stderr)

            labels = example["labels"]
            
            classes_in_example = set()
            for l in labels:
                 mapped = BioEvaluator.map_pii_to_ner_schema(l)
                 if "PER" in mapped: classes_in_example.add("PER")
                 elif "LOC" in mapped: classes_in_example.add("LOC")
                 elif "ORG" in mapped: classes_in_example.add("ORG")
            
            useful = False
            for c in classes_in_example:
                if counts[c] < MIN_PER_CLASS:
                    useful = True
            
            # If all minimums met, still fill up to 100 samples
            if not useful and len(selected_indices) < 100:
                 useful = True

            if useful:
                selected_indices.append(i)
                for c in classes_in_example:
                    counts[c] += 1
            
            # Stop if satisfied
            if all(c >= MIN_PER_CLASS for c in counts.values()) and len(selected_indices) >= 100:
                break
        
        print(f"{Fore.CYAN}Counts: {counts}")
        return shuffled.select(selected_indices)


class BioEvaluatorRunner(BioEvaluator): 
    def evaluate_model_on_dataset(self, model_name, dataset_name):
        print(f"\n{Fore.MAGENTA}>>> Evaluating Model: {model_name} on Dataset: {dataset_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
        except Exception as e:
            print(f"{Fore.RED}Failed to load model {model_name}: {e}")
            return

        id2label = model.config.id2label

        normalizer = DatasetNormalizerBIO(tokenizer)
        ds = normalizer.normalize(dataset_name)
        if ds is None or len(ds) == 0: 
            print(f"{Fore.RED}Dataset normalization failed or empty for {dataset_name}")
            return

        def tokenize_and_align(examples):
            tokenized_inputs = tokenizer(
                examples["tokens"], 
                truncation=True, 
                is_split_into_words=True
            )
            
            labels = []
            ref_labels = [] 
            
            for i, label_list in enumerate(examples["labels"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                ref_label_ids = []
                
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                        ref_label_ids.append("PAD")
                    elif word_idx != previous_word_idx:
                        if word_idx < len(label_list):
                            tag = str(label_list[word_idx])
                        else:
                            tag = "O"
                        
                        # Store RAW Ground Truth for report
                        # We want to report against the DATASET'S specific labels
                        ref_label_ids.append(tag) 
                        
                        # Map to Model's ID space (dummy)
                        label_ids.append(0)
                    else:
                        label_ids.append(-100)
                        ref_label_ids.append("SUBWORD")
                        
                    previous_word_idx = word_idx
                
                labels.append(label_ids)
                ref_labels.append(ref_label_ids)
            
            tokenized_inputs["labels"] = labels
            tokenized_inputs["reference_labels"] = ref_labels
            return tokenized_inputs

        tokenized_ds = ds.map(tokenize_and_align, batched=True)
        
        # Predict
        args = TrainingArguments(output_dir="./results_tmp", report_to="none", per_device_eval_batch_size=BATCH_SIZE)
        trainer = Trainer(model=model, args=args, tokenizer=tokenizer, data_collator=DataCollatorForTokenClassification(tokenizer))
        
        print(f"{Fore.BLUE}Running prediction...")
        predictions, _, _ = trainer.predict(tokenized_ds)
        preds = np.argmax(predictions, axis=2)
        
        # Flatten & Report
        true_labels_flat = []
        pred_labels_flat = []
        
        ref_all = tokenized_ds["reference_labels"]
        input_ids_all = tokenized_ds["input_ids"]
        
        print(f"\n{Fore.YELLOW}--- Sample Predictions (First 5) ---{Style.RESET_ALL}")
        
        for i in range(len(ref_all)):
            # Full Sentence Decode & Clean
            raw_text = tokenizer.decode(input_ids_all[i], skip_special_tokens=True)
            def clean_text(t):
                return t.replace(" ##", "").replace(" .", ".").replace(" ,", ",").replace(" ' ", "'").replace(" - ", "-")
            full_text = clean_text(raw_text)
            
            visible_tokens = []
            visible_true = []
            visible_pred = []
            
            for j in range(len(ref_all[i])):
                gt_tag = ref_all[i][j] # RAW tag (e.g. B-COMPANYNAME)
                
                if gt_tag not in ["PAD", "SUBWORD"]:
                    p_id = preds[i][j]
                    pred_label_raw = id2label.get(p_id, "O")
                    
                    # Logic: 
                    # 1. Map GT to Generic (e.g. B-COMPANYNAME -> B-ORG)
                    # 2. Map Pred to Generic (e.g. B-ORG -> B-ORG)
                    # 3. If Generic Types Match, Credit the Model with the Specific Label
                    
                    gt_generic = self.map_pii_to_ner_schema(gt_tag)
                    pred_generic = self.map_pii_to_ner_schema(pred_label_raw)
                    
                    final_pred = pred_generic
                    
                    # Check for "Credit": 
                    # If model predicted ORG and truth was COMPANYNAME (which maps to ORG),
                    # we say model predicted COMPANYNAME.
                    if gt_generic == pred_generic:
                        final_pred = gt_tag
                    
                    true_labels_flat.append(gt_tag)
                    pred_labels_flat.append(final_pred)
                    
                    token_id = input_ids_all[i][j]
                    token_str = tokenizer.decode([token_id])
                    
                    visible_tokens.append(token_str) # Keep raw
                    visible_true.append(gt_tag)
                    visible_pred.append(final_pred)
            
            if i < 5:
                def extract_entities(tokens, tags):
                    entities = []
                    current_ent = []
                    current_tag = None
                    for t, tag in zip(tokens, tags):
                        if str(tag).startswith("B-"):
                            if current_ent: 
                                joined = clean_text(" ".join(current_ent))
                                entities.append(f"{joined} ({current_tag})")
                            current_ent = [t]
                            current_tag = tag[2:]
                        elif str(tag).startswith("I-") and current_tag == tag[2:]:
                            current_ent.append(t)
                        else:
                            if current_ent: 
                                joined = clean_text(" ".join(current_ent))
                                entities.append(f"{joined} ({current_tag})")
                            current_ent = []
                            current_tag = None
                    if current_ent: 
                        joined = clean_text(" ".join(current_ent))
                        entities.append(f"{joined} ({current_tag})")
                    return entities

                true_ents = extract_entities(visible_tokens, visible_true)
                pred_ents = extract_entities(visible_tokens, visible_pred)
                
                print(f"{Fore.CYAN}Example {i+1}:")
                # Fix spacing in full_text display
                print(f"  Text: {full_text}") 
                print(f"  {Fore.GREEN}True Entities: {true_ents}")
                print(f"  {Fore.BLUE}Pred Entities: {pred_ents}{Style.RESET_ALL}")
                print("-" * 40)

        # Report
        labels_in_data = sorted(list(set(true_labels_flat + pred_labels_flat)))
        # Filter O and others
        target_names = [l for l in labels_in_data if l != "O" and l != "PAD"]
        
        print(f"\n{Fore.GREEN}Results for Model {model_name} on {dataset_name}:")
        print(classification_report(true_labels_flat, pred_labels_flat, labels=target_names, zero_division=0))
        print("-" * 60)

if __name__ == "__main__":
    evaluator = BioEvaluatorRunner()
    for model in BIO_MODELS:
        for dataset in BIO_DATASETS:
            evaluator.evaluate_model_on_dataset(model, dataset)
