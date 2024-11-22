import json
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

def clean_code(generated_code):
    # Remove Markdown formatting (triple backticks, "python", and any extra text before or after the code)
    if generated_code.startswith("```"):
        generated_code = re.sub(r"```(python)?", "", generated_code)  # Remove ```python or ```
        generated_code = generated_code.replace("```", "").strip()  # Remove closing ```
    # Remove any descriptive text that might be present before or after the actual code
    lines = generated_code.splitlines()
    # Extract only lines that are part of the code (ignoring any explanations, examples, etc.)
    code_lines = []
    inside_function = False
    for line in lines:
        if line.strip().startswith("def "):
            inside_function = True
        if inside_function:
            code_lines.append(line)
        if inside_function and line.strip() == "":  # Empty line after code signals end of function
            break
    generated_code = "\n".join(code_lines).strip()
    return generated_code

def generate_code(model, tokenizer, task_description, test_code):
    function_name = extract_function_name_from_code(code_mbpp)
    if not function_name:
        print(test_code)
        raise ValueError("Function name could not be extracted from code.")

    # Define prompt to get only the function code
    prompt = (
        f"Write only the Python function code for the following task:\n"
        f"{task_description}\n"
        f"The function must be named '{function_name}'.\n"
        f"Do not include any explanations, comments, or examples. Provide only the function code."
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

def extract_function_name_from_code(code_mbpp):
    """
    Extracts the function name and arguments from the provided code.
    
    Args:
        code_mbpp (str): A string containing the function code.
    
    Returns:
        str: The extracted function signature (name and arguments), or None if not found.
    """
    # Pattern to match the function name and arguments in a Python function definition
    pattern = r"def\s+(\w+)\s*\(([^)]*)\)"
    match = re.search(pattern, code_mbpp)
    if match:
        function_name = match.group(1)  # Extract the function name
        arguments = match.group(2)  # Extract the arguments
        return f"{function_name}({arguments})"  # Return the function signature
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
        correct_code = example["code"]
        # Generate code
        generated_code = generate_code(model, tokenizer, task_description, correct_code)

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
