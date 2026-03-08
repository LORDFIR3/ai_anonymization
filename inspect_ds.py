from datasets import load_dataset
dataset = load_dataset("nvidia/Nemotron-PII")
example = dataset["train"][0]
print(f"Type of spans: {type(example['spans'])}")
print(f"Spans content: {example['spans']}")
if isinstance(example['spans'], str):
    import json
    try:
        spans_json = json.loads(example['spans'])
        print(f"Parsed spans type: {type(spans_json)}")
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        import ast
        spans_ast = ast.literal_eval(example['spans'])
        print(f"Parsed ast type: {type(spans_ast)}")
