import numpy as np
import pandas as pd
import json
import ast
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

SPANS_MODELS = [
    "OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1", 
    "gravitee-io/bert-small-pii-detection", 
]

SPANS_DATASETS = [
    "nvidia/Nemotron-PII",
    "gretelai/synthetic_pii_finance_multilingual"
]

class SpansEvaluator:
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
            
        # ORG (Prioritize over NAME to strict match COMPANYNAME)
        if any(x == name for x in ["ORG", "COMPANY", "COMPANYNAME", "BANK", "ISSUER", "UNIVERSITY"]):
            return f"{prefix}ORG"
        if "COMPANY" in name or "ORG" in name: 
            return f"{prefix}ORG"

        # PER
        if any(x in name for x in ["PER", "FIRSTNAME", "LASTNAME", "MIDDLENAME", "USERNAME", "NAME", "USER", "ACCOUNT"]):
            if "NUMBER" in name: return label_raw 
            return f"{prefix}PER"

        # LOC
        if any(x in name for x in ["LOC", "CITY", "STATE", "COUNTRY", "ADDRESS", "ZIP", "GPS", "STREET", "BUILDING", "COUNTY"]):
            return f"{prefix}LOC"
                
        return label_raw

class DatasetNormalizerSpans:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def normalize(self, dataset_name, split="train"):
        print(f"{Fore.CYAN}--- Processing SPANS Dataset: {dataset_name} ---")
        
        try:
            ds = load_dataset(dataset_name, split=split)
        except Exception as e:
            print(f"{Fore.RED}Could not load {dataset_name}: {e}")
            return None
        
        text_col, spans_col = None, None
        if "nvidia" in dataset_name:
            text_col, spans_col = "text", "spans"
        elif "gretelai" in dataset_name:
            cols = ds.column_names
            text_col = "generated_text" if "generated_text" in cols else "text"
            spans_col = "pii_spans" if "pii_spans" in cols else "spans"
        else:
            print(f"{Fore.RED}Unknown structure for {dataset_name}")
            return None

        print(f"{Fore.YELLOW}Performing stratified sampling (Target: {MIN_PER_CLASS} per class)...")
        sampled_ds = self._stratified_sample(ds, spans_col)
        print(f"{Fore.GREEN}Selected {len(sampled_ds)} samples.")
        
        return self._normalize_spans(sampled_ds, text_col, spans_col)


    def _stratified_sample(self, dataset, spans_col):
        shuffled = dataset.shuffle(seed=SEED)
        
        selected_indices = []
        counts = {"PER": 0, "LOC": 0, "ORG": 0}
        
        def parse_spans_safe(spans_raw):
            if isinstance(spans_raw, str):
                try: 
                    return json.loads(spans_raw)
                except:
                    try: 
                        return ast.literal_eval(spans_raw)
                    except: 
                        return []
            return spans_raw if isinstance(spans_raw, list) else []

        def get_classes(spans):
            found = set()
            for s in spans:
                l_str = s.get('label', s.get('entity_type', ''))
                # Use shared mapping logic
                mapped = SpansEvaluator.map_pii_to_ner_schema(l_str)
                if "PER" in mapped: found.add("PER")
                elif "LOC" in mapped: found.add("LOC")
                elif "ORG" in mapped: found.add("ORG")
            return found

        print("Sampling iteration...", file=sys.stderr)

        for i, example in enumerate(shuffled):
            if len(selected_indices) >= MAX_SAMPLES:
                break
            
            if i > 0 and i % 10000 == 0:
                 print(f"Scanned {i} examples... Found: {counts}", file=sys.stderr)
                
            spans_raw = example.get(spans_col)
            spans = parse_spans_safe(spans_raw)
            classes_in_example = get_classes(spans)
            
            useful = False
            for c in classes_in_example:
                if counts[c] < MIN_PER_CLASS:
                    useful = True
            
            if not useful and len(selected_indices) < 100:
                 useful = True
            
            if useful:
                selected_indices.append(i)
                for c in classes_in_example:
                    counts[c] += 1
            
            if all(c >= MIN_PER_CLASS for c in counts.values()) and len(selected_indices) >= 100:
                 break
        
        print(f"{Fore.CYAN}Counts: {counts}")
        return shuffled.select(selected_indices)


    def _normalize_spans(self, dataset, text_col, spans_col):
        def process(example):
            text = example.get(text_col, "")
            spans_raw = example.get(spans_col, [])
            
            if isinstance(spans_raw, str):
                try: spans = json.loads(spans_raw)
                except:
                    try: spans = ast.literal_eval(spans_raw)
                    except: spans = [] 
            else:
                spans = spans_raw
            
            # Tokenize with strict mapping to max length
            # We want to perform BIO tagging on the subwords directly.
            tokenized = self.tokenizer(
                text, 
                return_offsets_mapping=True, 
                truncation=True, 
                max_length=512,
                padding="max_length" # Optional, consistency
            )
            
            # These are now ready for model
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            offsets = tokenized["offset_mapping"]
            
            # Create BIO labels for these subwords
            labels = ["O"] * len(input_ids)
            
            clean_spans = []
            if isinstance(spans, list):
                for s in spans:
                    start = s.get('start', s.get('start_position'))
                    end = s.get('end', s.get('end_position'))
                    label = s.get('label', s.get('entity_type'))
                    if start is not None and end is not None and label:
                         clean_spans.append((int(start), int(end), label))
            
            for (start_char, end_char, label_str) in clean_spans:
                for idx, (token_start, token_end) in enumerate(offsets):
                    if token_start == 0 and token_end == 0: continue 
                    
                    # Strictly inside span? intersection?
                    # Generally: if subword overlaps significantly or starts inside
                    if token_start >= start_char and token_end <= end_char:
                        if token_start == start_char or (token_start > start_char and labels[idx-1] == "O"):
                             labels[idx] = f"B-{label_str}"
                        else:
                             if labels[idx] == "O": 
                                 if idx > 0 and (str(labels[idx-1]).endswith(label_str)):
                                     labels[idx] = f"I-{label_str}"
                                 else:
                                     labels[idx] = f"B-{label_str}"

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "bio_tags": labels # Keep strings here, map later or return ids if we had label2id
            }

        return dataset.map(process, remove_columns=dataset.column_names)


class SpansEvaluatorRunner(SpansEvaluator):
    def evaluate_model_on_dataset(self, model_name, dataset_name):
        print(f"\n{Fore.MAGENTA}>>> Evaluating Model: {model_name} on Dataset: {dataset_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
        except Exception as e:
            print(f"{Fore.RED}Failed to load model {model_name}: {e}")
            return

        label2id = model.config.label2id
        id2label = model.config.id2label

        normalizer = DatasetNormalizerSpans(tokenizer)
        ds = normalizer.normalize(dataset_name)
        if ds is None or len(ds) == 0: return

        # Dataset already has input_ids. We just need to add 'labels' (ints) from 'bio_tags' (strings)
        def prepare_for_model(example):
            bio_tags = example["bio_tags"]
            label_ids = []
            ref_labels = [] # Ground Truth Mapped
            
            for tag in bio_tags:
                # Map to Int
                # Use generic "O" id if not found (or model specific O)
                if tag in label2id:
                    label_ids.append(label2id[tag])
                else:
                    # Try to map B-PER -> B-PER etc if schema matches? 
                    # If model uses specific schema, we might miss.
                    # Simpler: If not in label2id, assume 'O' (ID for O)
                    # Find 'O' ID
                    o_id = label2id.get("O", 0) 
                    label_ids.append(o_id)

                # Reference for Report (Generic Mapping)
                # Store RAW tag for Native Reporting (e.g. B-COMPANYNAME)
                ref_labels.append(tag)

            return {
                "input_ids": example["input_ids"],
                "attention_mask": example["attention_mask"],
                "labels": label_ids,
                "reference_labels": ref_labels
            }

        tokenized_ds = ds.map(prepare_for_model)
        
        args = TrainingArguments(output_dir="./results_tmp", report_to="none", per_device_eval_batch_size=BATCH_SIZE)
        trainer = Trainer(model=model, args=args, tokenizer=tokenizer, data_collator=DataCollatorForTokenClassification(tokenizer))
        
        print(f"{Fore.BLUE}Running prediction...")
        predictions, _, _ = trainer.predict(tokenized_ds)
        preds = np.argmax(predictions, axis=2)
        
        true_labels_flat = []
        pred_labels_flat = []
        
        ref_all = tokenized_ds["reference_labels"]
        input_ids_all = tokenized_ds["input_ids"]
        
        print(f"\n{Fore.YELLOW}--- Sample Predictions (First 5) ---{Style.RESET_ALL}")
        
        for i in range(len(ref_all)):
            # Full Sentence Decode & Clean
            def clean_text(t):
                return t.replace(" ##", "").replace(" .", ".").replace(" ,", ",").replace(" ' ", "'").replace(" - ", "-")

            raw_text = tokenizer.decode(input_ids_all[i], skip_special_tokens=True)
            full_text = clean_text(raw_text)
            
            visible_tokens = []
            visible_true = []
            visible_pred = []
            
            # Iterate tokens
            for j in range(len(ref_all[i])):
                # Check attention mask if available, or special token ID
                token_id = input_ids_all[i][j]
                
                # Retrieve GT (Raw)
                gt = ref_all[i][j]
                
                # Skip Special Tokens/Padding for Metric Calculation
                if token_id in iter(tokenizer.all_special_ids) or token_id == 0: 
                    continue

                p_id = preds[i][j]
                pred_label_raw = id2label.get(p_id, "O")
                
                # Logic: Native Label Reporting
                gt_generic = self.map_pii_to_ner_schema(gt)
                pred_generic = self.map_pii_to_ner_schema(pred_label_raw)
                
                final_pred = pred_generic
                # Credit specific class if generic type matches
                if gt_generic == pred_generic:
                    final_pred = gt
                    
                true_labels_flat.append(gt)
                pred_labels_flat.append(final_pred)
                
                # For visualization
                token_str = tokenizer.decode([token_id])
                visible_tokens.append(token_str) # Keep raw with ## here
                visible_true.append(gt)
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
                print(f"  Text: {full_text}")
                print(f"  {Fore.GREEN}True Entities: {true_ents}")
                print(f"  {Fore.BLUE}Pred Entities: {pred_ents}{Style.RESET_ALL}")
                print("-" * 40)

        labels_in_data = sorted(list(set(true_labels_flat + pred_labels_flat)))
        target_names = [l for l in labels_in_data if l != "O" and l != "PAD"]
        
        print(f"\n{Fore.GREEN}Results for Model {model_name} on {dataset_name}:")
        print(classification_report(true_labels_flat, pred_labels_flat, labels=target_names, zero_division=0))
        print("-" * 60)

if __name__ == "__main__":
    evaluator = SpansEvaluatorRunner()
    for model in SPANS_MODELS:
        for dataset in SPANS_DATASETS:
            evaluator.evaluate_model_on_dataset(model, dataset)
