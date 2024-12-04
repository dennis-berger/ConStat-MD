import json
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import unittest
import io
import sys
import argparse

def clean_code(generated_code):
    if generated_code.startswith("```"):
        generated_code = re.sub(r"```(python)?", "", generated_code)
        generated_code = generated_code.replace("```", "").strip()
    lines = generated_code.splitlines()
    code_lines = []
    inside_function = False
    for line in lines:
        if line.strip().startswith("def "):
            inside_function = True
        if inside_function:
            code_lines.append(line)
        if inside_function and line.strip() == "":
            break
    generated_code = "\n".join(code_lines).strip()
    return generated_code

def generate_code(model, tokenizer, task_description, test_code, dataset_name):
    if dataset_name == "openai_humaneval":
        # For HumanEval, use the full prompt directly without extracting function name
        prompt = (
            f"Write only the Python function code for the following task:\n"
            f"{task_description}\n"
            f"Include any necessary import statements. Do not include any explanations, comments, or examples."
        )
    else:
        function_name = extract_function_name_from_code(test_code)
        if not function_name:
            print(task_description)
            raise ValueError("Function name could not be extracted from code.")

        prompt = (
            f"Write only the Python function code for the following task:\n"
            f"{task_description}\n"
            f"The function must be named '{function_name}'.\n"
            f"Include any necessary import statements. Do not include any explanations, comments, or examples."
        )

    messages = [
        {"role": "system", "content": "You are a coding assistant that provides only Python code snippets."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    clean_generated_code = clean_code(response)
    return clean_generated_code

def extract_function_name_from_code(test_code):
    pattern = r"def\s+(\w+)\s*\(([^)]*)\)"
    match = re.search(pattern, test_code)
    if match:
        function_name = match.group(1)
        arguments = match.group(2)
        return f"{function_name}({arguments})"
    return None

def test_generated_code(generated_code, test_code, dataset_name):
    test_results = []
    all_tests_passed = True
    try:
        exec(generated_code, globals())
        if dataset_name == "openai_humaneval":
            # For HumanEval, extract the check function and run it with the generated code
            exec(test_code, globals())
            check_function = globals().get("check")
            if check_function:
                try:
                    check_function(globals()[generated_code.split("(")[0]])
                    test_results.append(True)
                except AssertionError:
                    test_results.append(False)
                    all_tests_passed = False
                except Exception as e:
                    print(f"Runtime error during check function execution: {e}")
                    test_results.append(False)
                    all_tests_passed = False
            else:
                print("No check function found in test code.")
                all_tests_passed = False
                test_results = [False]
        else:
            # For other datasets, execute each test case
            for test_case in test_code:
                try:
                    exec(test_case, globals())
                    test_results.append(True)
                except AssertionError:
                    test_results.append(False)
                    all_tests_passed = False
    except Exception as e:
        print(f"Compilation or execution error: {e}")
        all_tests_passed = False
        test_results = [False] * len(test_code)
    return test_results, all_tests_passed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Name of the model to run')
    args = parser.parse_args()
    model_name = args.model
    models = [
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-3.2-3B-Instruct"
    ]
    if model_name not in models:
        raise ValueError(f"Invalid model name: {model_name}")
    
    datasets_info = [
        ("google-research-datasets/mbpp", "sanitized", "train"),
        ("google-research-datasets/mbpp", "sanitized", "test"),
        ("openai_humaneval", None, "test")
    ]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for dataset_name, config, split in datasets_info:
        print("Testing dataset " + model_name + " on " + dataset_name)
        dataset = load_dataset(dataset_name, config, split=split)
        results = []
        for example in dataset:
            if dataset_name == "openai_humaneval":
                task_description = example["prompt"]
                test_code = example["test"]
            else:
                task_description = example["prompt"]
                test_code = example["test_list"]
            correct_code = example["code"] if "code" in example else example["canonical_solution"]
            generated_code = generate_code(model, tokenizer, task_description, correct_code, dataset_name)
            test_results, all_tests_passed = test_generated_code(generated_code, test_code, dataset_name)
            result = {
                "task_description": task_description,
                "generated_code": generated_code,
                "test_code": test_code,
                "test_results": test_results,
                "all_tests_passed": all_tests_passed
            }
            results.append(result)

        file_name = f"{model_name.replace('/', '_')}_{dataset_name.replace('/', '_')}_{split}_results.json"
        with open(file_name, "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
