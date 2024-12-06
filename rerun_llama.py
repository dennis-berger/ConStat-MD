import json
import os
import re


# Directory paths
input_folder = "./results"
output_folder = "./processed_results"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

def extract_function_code(code):
    """
    Extract the function code from the generated_code field.
    Handles functions spanning multiple lines with proper indentation.
    """
    match = re.search(r"(def\s+\w+\s*\(.*?\):(?:\n(?:\s{4}|\t).+)+)", code)
    if match:
        return match.group(1)
    elif code.strip().startswith("def"):
        return code.strip()
    return None

def extract_humaneval_tests_as_array(test_code):
    """
    Extract test cases from the 'check(candidate)' function as an array.
    Handles varied formats, including additional imports and edge cases.
    """
    test_lines = []
    try:
        # Locate the "def check(candidate)" block
        match = re.search(r"def check\(candidate\):\n((?:\s{4,}.*\n)+)", test_code)
        if match:
            raw_tests = match.group(1).strip()
            # Split and clean lines, replacing 'candidate' with 'func'
            for line in raw_tests.splitlines():
                cleaned_line = re.sub(r"\bcandidate\b", "func", line.strip())
                if cleaned_line.startswith("assert"):  # Only keep assert statements
                    test_lines.append(cleaned_line)
    except Exception as e:
        print(f"Error processing test_code: {e}")
    return test_lines

def run_test_cases(func_code, test_cases):
    """
    Execute the test cases and return the results.
    Dynamically defines the function and executes assertions.
    Includes debugging outputs for verification.
    """
    test_results = []
    try:
        # Debugging: Validate and define the function dynamically
        
        exec(func_code, globals())
        globals()['func'] = eval(func_code.split('(')[0].split()[-1])  # Extract function name

        for test in test_cases:
            try:
                exec(test)
                test_results.append(True)
            except Exception as e:
                print(f"Test failed: {test}\nError: {e}")
                test_results.append(False)
    except Exception as e:
        print(f"Error defining function:\n{func_code}\nError: {e}")
        test_results = [False] * len(test_cases)
    return test_results

# Process files
for file_name in os.listdir(input_folder):
    if ("mbpp" in file_name or "humaneval" in file_name) and file_name.endswith(".json"):
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, file_name)

        with open(input_file_path, "r") as file:
            data = json.load(file)

        for entry in data:
            # Extract and clean function code
            raw_code = entry.get("generated_code", "")
            clean_code = extract_function_code(raw_code)

            # Extract and process test cases
            test_code = entry.get("test_code", "")
            if "humaneval" in file_name:
                test_cases = extract_humaneval_tests_as_array(test_code)
            else:
                test_cases = entry.get("test_code", [])

            # Update test_code as an array in the JSON structure
            entry["test_code"] = test_cases

            # Run the tests
            if clean_code:
                entry["generated_code"] = clean_code
                entry["test_results"] = run_test_cases(clean_code, test_cases)
                entry["all_tests_passed"] = all(entry["test_results"])
            else:
                entry["generated_code"] = "Invalid function code"
                entry["test_results"] = [False] * len(test_cases)
                entry["all_tests_passed"] = False

        # Save the results
        with open(output_file_path, "w") as file:
            json.dump(data, file, indent=4)

print(f"Processing complete. Updated files are saved in {output_folder}")
