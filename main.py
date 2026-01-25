import numpy as np
import pandas as pd
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report
from collections import Counter
from colorama import Fore, Style, init

init(autoreset=True)

# --- CONFIGURATION ---
DATASET_NAME = "ai4privacy/pii-masking-200k"
MODEL_CHECKPOINT = "dslim/bert-base-NER" 
BATCH_SIZE = 16
SAMPLE_SIZE = 4000  # Load 4000 rows, then split 50/50

class PIIEvaluator:
    def __init__(self):
        print(f"{Fore.CYAN}--- Step 1: Loading & Splitting Dataset ---")
        
        # 1. Load a slice of the TRAIN split (since validation doesn't exist)
        # We load 4000 samples to keep it fast, but you can increase this.
        full_dataset = load_dataset(DATASET_NAME, split=f"train[:{SAMPLE_SIZE}]")
        
        # 2. Split it manually (50% Train (unused here), 50% Validation)
        # We use a seed so the split is the same every time you run it.
        splits = full_dataset.train_test_split(test_size=0.5, seed=42)
        self.dataset = splits['test']  # We use the 'test' half as our validation set
        
        print(f"{Fore.GREEN}✔ Loaded {len(full_dataset)} rows.")
        print(f"{Fore.GREEN}✔ Created Validation Split: {len(self.dataset)} rows.")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        self.model = AutoModelForTokenClassification.from_pretrained(MODEL_CHECKPOINT)
        
        # Model's expected ID map (e.g., {'B-PER': 1, 'O': 0})
        self.model_label2id = self.model.config.label2id
        self.model_id2label = self.model.config.id2label

        # DATASET'S actual ID map (If available)
        self.ds_id2label = {}
        try:
            # Try to see if it's a ClassLabel feature
            features = self.dataset.features['mbert_bio_labels']
            if hasattr(features, 'feature') and hasattr(features.feature, 'names'):
                self.ds_features = features.feature
                self.ds_id2label = {i: name for i, name in enumerate(self.ds_features.names)}
                print(f"{Fore.GREEN}✔ Found {len(self.ds_id2label)} unique classes in the dataset metadata.")
            else:
                 print(f"{Fore.YELLOW}⚠ Dataset labels appear to be raw strings (no ClassLabel metadata found).")
        except Exception as e:
            print(f"{Fore.RED}Could not extract dataset class names: {e}")

    def analyze_dataset(self):
        print(f"\n{Fore.CYAN}--- Step 2: TRUE Dataset Statistics ---")
        total_tokens = 0
        label_counts = Counter()
        
        # We iterate through the raw labels
        for output in self.dataset:
            labels = output['mbert_bio_labels']
            for l in labels:
                total_tokens += 1
                # Handle both Integer IDs and String Names
                if isinstance(l, str):
                    label_name = l
                else:
                    label_name = self.ds_id2label.get(l, f"Unknown_ID_{l}")
                
                label_counts[label_name] += 1
                    
        stats_data = []
        for label_name, count in label_counts.items():
            # Filter out 'O' and special tokens to reduce noise in the table
            if label_name not in ["O", "0", "Unknown_ID_-100"]:
                stats_data.append({"Entity Type": label_name, "Count": count, "Freq": f"{(count/total_tokens)*100:.4f}%"})
            
        df = pd.DataFrame(stats_data).sort_values(by="Count", ascending=False)
        
        print(f"\n{Fore.YELLOW}Table 1: True Distribution of PII Entities (Validation Set)")
        # Show all entities to fully describe dataset
        print(df.to_string(index=False)) 
        print(f"\n{Fore.MAGENTA}Total unique entity types found: {len(stats_data)}")

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["mbert_text_tokens"], 
            truncation=True, 
            is_split_into_words=True 
        )

        labels = []
        reference_labels = [] 
        for i, label_list in enumerate(examples["mbert_bio_labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            ref_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                    ref_ids.append("PAD") # Marker for padding
                elif word_idx != previous_word_idx:
                    # 1. Get the TRUE label name
                    original_val = label_list[word_idx]
                    if isinstance(original_val, str):
                        original_name = original_val
                    else:
                        original_name = self.ds_id2label.get(original_val, "O")
                    
                    # Store true original NAME for evaluation comparison
                    ref_ids.append(original_name)

                    # 2. Check if the MODEL knows this label
                    # The model only knows: B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, O
                    # If dataset has "B-EMAIL", model doesn't know it -> Map to "O" (0)
                    model_id = self.model_label2id.get(original_name, 0)
                    
                    label_ids.append(model_id)
                else:
                    label_ids.append(-100)
                    ref_ids.append("SUBWORD") # Marker for subwords
                previous_word_idx = word_idx
            
            labels.append(label_ids)
            reference_labels.append(ref_ids)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["reference_labels"] = reference_labels
        return tokenized_inputs

    def map_pii_to_ner_schema(self, label):
        """
        Maps a granular PII label (e.g., 'B-FIRSTNAME') to a generic NER label (e.g., 'B-PER')
        supported by the dslim/bert-base-NER model.
        """
        if label.startswith("B-") or label.startswith("I-"):
            prefix, name = label[:2], label[2:]
            
            # Person-related
            if name in ["FIRSTNAME", "LASTNAME", "MIDDLENAME", "USERNAME", "ACCOUNTNAME"]:
                return f"{prefix}PER"
            
            # Location-related
            if name in ["CITY", "STATE", "COUNTY", "ZIPCODE", "STREET", "BUILDINGNUMBER", 
                        "SECONDARYADDRESS", "ORDINALDIRECTION", "NEARBYGPSCOORDINATE"]:
                return f"{prefix}LOC"
            
            # Organization-related
            if name in ["COMPANYNAME", "CREDITCARDISSUER"]:
                return f"{prefix}ORG"

        # Return original if no clear mapping (e.g., EMAIL, PASSWORD, BITCOINADDRESS)
        # These will essentially remain mistakes for the baseline model, which is correct.
        return label

    def evaluate_model(self):
        print(f"\n{Fore.CYAN}--- Step 3: Evaluation (Baseline) ---")
        
        tokenized_dataset = self.dataset.map(self.tokenize_and_align_labels, batched=True)
        
        args = TrainingArguments(
            output_dir="./results",
            per_device_eval_batch_size=BATCH_SIZE,
            report_to="none"
        )
        
        trainer = Trainer(
            model=self.model,
            args=args,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForTokenClassification(self.tokenizer)
        )
        
        print("Running Inference...")
        predictions, _, _ = trainer.predict(tokenized_dataset)
        preds = np.argmax(predictions, axis=2)

        true_labels_flat = []
        true_preds_flat = []

        print(f"\n{Fore.YELLOW}--- Sample Predictions (First 5) ---")
        
        # We need to access the 'reference_labels' from the tokenized dataset
        # because the trainer.predict output 'labels' usually corresponds to the model's mapped labels
        ref_labels_all = tokenized_dataset["reference_labels"]
        
        for i in range(len(ref_labels_all)):
            ref_ids = ref_labels_all[i]
            pred_ids = preds[i] # These are model IDs
            
            sample_tokens = []
            sample_true = []
            sample_pred = []
            
            for j in range(len(ref_ids)):
                # Filter out PAD and SUBWORD for the report
                if ref_ids[j] not in ["PAD", "SUBWORD"]:
                    # 1. True Label: It is already the Name string
                    t_lbl = ref_ids[j]
                    true_labels_flat.append(t_lbl)
                    
                    # 2. Predicted Label: Use MODEL map (only standard NER)
                    p_lbl = self.model_id2label.get(pred_ids[j], "O")
                    true_preds_flat.append(p_lbl)
                    
                    if i < 5:
                        sample_true.append(t_lbl)
                        sample_pred.append(p_lbl)
            
            if i < 5:
                # Filter O for cleaner view
                vis_true = [t for t in sample_true if t != "O"]
                vis_pred = [p for p in sample_pred if p != "O"]
                
                print(f"\nSample {i+1}:")
                if not vis_true and not vis_pred:
                    print("  (No entities found in this sample)")
                else:
                    print(f"  TRUE Data: {vis_true}")
                    print(f"  MODEL Pred: {vis_pred}")

        print(f"\n{Fore.YELLOW}--- Classification Report 1: Strict (Specific PII Classes) ---")
        # We use a large set of labels to ensure even 0-score classes appear
        unique_true = sorted(list(set(true_labels_flat)))
        report = classification_report(true_labels_flat, true_preds_flat, labels=unique_true, zero_division=0)
        print(report)

        # --- MAPPED EVALUATION ---
        print(f"\n{Fore.CYAN}--- Optimization: Generating Mapped Report for General NER Overlap ---")
        
        # Convert the TRUE detailed labels to Generic NER labels (FIRSTNAME -> PER)
        # to see if the model at least detects the *type* of entity.
        mapped_true_labels = [self.map_pii_to_ner_schema(l) for l in true_labels_flat]
        
        # Filter for labels that actually exist in the model's vocabulary (plus 'O')
        valid_model_labels = set(self.model_id2label.values())
        
        # We only want to report on classes that are relevant to the MODEL (PER, LOC, ORG, MISC)
        # But we must include the mapped dataset labels even if they don't match perfect model keys
        # to show what is still missed (e.g. B-EMAIL maps to B-EMAIL, not supported).
        
        unique_mapped = sorted(list(set(mapped_true_labels)))
        
        print(f"{Fore.YELLOW}--- Classification Report 2: Generalized (Mapped to PER/LOC/ORG) ---")
        report_mapped = classification_report(mapped_true_labels, true_preds_flat, labels=unique_mapped, zero_division=0)
        print(report_mapped)
        
        print("\n" + "="*60)
        print("\n" + "="*60)
        print(f"{Fore.MAGENTA}INTERPRETING THE RESULTS:")
        print("1. STRICT REPORT: Shows 0.00 for mostly everything because 'B-EMAIL' != 'O'.")
        print("   This proves the baseline model does not natively understand PII specific tags.")
        print("-" * 30)
        print("2. GENERALIZED REPORT: Shows higher scores where concepts overlap.")
        print("   Example: 'B-FIRSTNAME' mapped to 'B-PER' matches the model's 'B-PER' prediction.")
        print(f"{Fore.GREEN}   If you see decent scores here (e.g. on PER/LOC), the model IS working correctly")
        print("   for generic entities, just not precise PII types like 'Emails' or 'Crypto Addresses'.")
        print("="*60)

# --- EXECUTION ---
runner = PIIEvaluator()
runner.analyze_dataset()
runner.evaluate_model()
