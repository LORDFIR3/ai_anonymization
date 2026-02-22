import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import classification_report

def get_columns(dataset):
    cols = dataset.column_names
    tokens_col, labels_col = None, None
    if "tokens" in cols: tokens_col = "tokens"
    elif "mbert_text_tokens" in cols: tokens_col = "mbert_text_tokens"
    elif "tokenised_text" in cols: tokens_col = "tokenised_text"
        
    if "labels" in cols: labels_col = "labels"
    elif "bio_labels" in cols: labels_col = "bio_labels"
    elif "mbert_bio_labels" in cols: labels_col = "mbert_bio_labels"
    elif "ner_tags" in cols: labels_col = "ner_tags"
    return tokens_col, labels_col

def evaluate_model(model_id, dataset_id, num_samples=1000):
    print(f"\n\n{'='*80}")
    print(f"Evaluating Model: '{model_id}'")
    print(f"On Dataset:       '{dataset_id}'")
    print(f"{'='*80}")
    
    try:
        ds = load_dataset(dataset_id, split="train")
    except Exception as e:
        print(f"Error loading dataset {dataset_id}: {e}")
        return
        
    # Take 1000 samples
    samples_count = min(num_samples, len(ds))
    ds = ds.shuffle(seed=42).select(range(samples_count))
    print(f"Selected {samples_count} samples.")
    
    tokens_col, labels_col = get_columns(ds)
    if not tokens_col or not labels_col:
        print(f"Could not automatically determine tokens and labels columns in {ds.column_names}")
        return
        
    print(f"Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    id2label = model.config.id2label
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    true_labels_flat = []
    pred_labels_flat = []
    
    for idx, item in enumerate(ds):
        tokens = item[tokens_col]
        labels = item[labels_col]
        # Ensure labels are string representations
        labels = [str(l) for l in labels]
        
        inputs = tokenizer(
            tokens, 
            is_split_into_words=True, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        word_ids = inputs.word_ids()
        
        # move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
        
        pred_for_token = ["O"] * len(tokens)
        previous_word_idx = None
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            elif word_idx != previous_word_idx:
                if word_idx < len(tokens):
                    p_id = predictions[i]
                    pred_label = id2label.get(p_id, "O")
                    pred_for_token[word_idx] = pred_label
                    
                    gt_tag = labels[word_idx]
                    true_labels_flat.append(gt_tag)
                    pred_labels_flat.append(pred_label)
                    
            previous_word_idx = word_idx
            
        # Visualization
        if idx < 3:
            print(f"\n--- Example {idx + 1} ---")
            
            raw_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            def clean_text(t):
                return t.replace(" ##", "").replace(" .", ".").replace(" ,", ",").replace(" ' ", "'").replace(" - ", "-")
            
            full_text = clean_text(raw_text)
            print(f"Original Text: {full_text}")
            
            def extract_entities(tokens_list, tags_list):
                entities = []
                current_ent = []
                current_tag = None
                for t, tag in zip(tokens_list, tags_list):
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
                
            true_ents = extract_entities(tokens, labels)
            pred_ents = extract_entities(tokens, pred_for_token)
            
            print(f"\nTrue Entities:")
            if true_ents:
                for e in true_ents: print(f"  {e}")
            else:
                print("  None")
                
            print(f"\nPred Entities:")
            if pred_ents:
                for e in pred_ents: print(f"  {e}")
            else:
                print("  None")
            print("-" * 40)
            
    # Report classes dynamically
    labels_in_data = sorted(list(set(true_labels_flat + pred_labels_flat)))
    target_names = [l for l in labels_in_data if l not in ["O", "PAD", "-100", "SUBWORD"]]
    
    print(f"\nClassification Report ({samples_count} samples):")
    if not target_names:
        print("No entities found in true or predicted labels.")
    else:
        print(classification_report(true_labels_flat, pred_labels_flat, labels=target_names, zero_division=0))

if __name__ == "__main__":
    # Extracted BIO-compatible models from modelList.txt
    PAIRS = [
        ("Isotonic/distilbert_finetuned_ai4privacy_v2", "Isotonic/pii-masking-200k"),
        ("Isotonic/deberta-v3-base_finetuned_ai4privacy_v2", "ai4privacy/pii-masking-200k")
    ]
    for m, d in PAIRS:
        evaluate_model(m, d, num_samples=1000)
