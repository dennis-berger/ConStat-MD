import json
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

def clean_code(generated_code):
    # Remove Markdown formatting (triple backticks and "python")
    if generated_code.startswith("```"):
        generated_code = re.sub(r"```(python)?", "", generated_code)  # Remove ```python or ```
        generated_code = generated_code.replace("```", "").strip()  # Remove closing ```
    return generated_code

def generate_code(model, tokenizer, task_description, test_code):
    function_name = extract_function_name_from_tests(test_code)
    if not function_name:
        print(test_code)
        raise ValueError("Function name could not be extracted from test cases.")

    # Define prompt to get only the function code
    prompt = (
        f"Write only the Python function code for the following task:\n"
        f"{task_description}\n"
        f"The function must be named '{function_name}'."
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

def extract_function_name_from_tests(test_cases):
    """
    Extracts the function name from the test cases.
    
    Args:
        test_cases (list): A list of test case strings.
    
    Returns:
        str: The extracted function name, or None if not found.
    """
    # Pattern to match function names (e.g., "function_name" in "assert function_name(...)")
    pattern = r"assert\s+(\w+)\s*\("
    
    for test in test_cases:
        match = re.search(pattern, test)
        if match:
            return match.group(1)  # Return the function name
    return None  # Return None if no function name is found


def test_generated_code(generated_code, test_cases):
    # Dictionary to store the results
    test_results = []
    all_tests_passed = True

    try:
        # Compile and execute the generated code
        exec(generated_code, globals())

        # Run each test case
        for test_code in test_cases:
            try:
                # Execute the test case
                exec(test_code, globals())
                test_results.append(True)
            except AssertionError:
                # If assertion fails, mark as False
                test_results.append(False)
                all_tests_passed = False
    except Exception as e:
        # If the code itself failed to compile, mark all as False
        print(f"Compilation or execution error: {e}")
        all_tests_passed = False
        test_results = [False] * len(test_cases)
    
    return test_results, all_tests_passed

def main():
    # Load dataset
    # dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    dataset = Dataset.load_from_disk("rephrased_dataset")
    # Initialize model and tokenizer
    # try meta-llama/Llama-2-7b-chat-hf
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    results = []

    # Loop over dataset examples
    for example in dataset:
        task_description = example["prompt"]
        test_code = example["test_list"]
        code_mbpp = example["code"]
        # Generate code
        generated_code = generate_code(model, tokenizer, task_description, test_code)

        # Run tests on the generated code
        test_results, all_tests_passed = test_generated_code(generated_code, test_code)

        # Record results
        result = {
            "task_description": task_description,
            "generated_code": generated_code,
            "test_code": test_code,
            "test_results": test_results,
            "all_tests_passed": all_tests_passed
        }
        results.append(result)
    # Save results to a JSON file
    with open("code_generation_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
