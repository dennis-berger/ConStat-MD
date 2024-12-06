import os
import json
import numpy as np
from constat import ConStat

# Directory containing the JSON files
data_dir = "./processed_results/"

# Define file keywords
mbpp_test_keyword = "google-research-datasets_mbpp_test_results.json"
mbpp_train_keyword = "google-research-datasets_mbpp_train_results.json"
humaneval_keyword = "openai_humaneval_test_results.json"

# Dynamically find files with a specific keyword
def find_files_with_keyword(directory, keyword):
    """Find all files in a directory matching a specific keyword."""
    return [
        os.path.join(directory, file_name)
        for file_name in os.listdir(directory)
        if keyword in file_name
    ]

# Find all MBPP test files
mbpp_test_files = find_files_with_keyword(data_dir, mbpp_test_keyword)

# Match MBPP test file to its corresponding model
def match_test_file(model_file, test_files):
    """Find the corresponding MBPP test file for a given model file."""
    model_name = model_file.split("_")[0]
    for test_file in test_files:
        if model_name in test_file:
            return test_file
    raise FileNotFoundError(f"No matching MBPP test file found for {model_file}")

# Process accuracies from benchmark data
def calculate_accuracies(data):
    return np.array([1 if entry["all_tests_passed"] else 0 for entry in data])

# Initialize ConStat
constat = ConStat(
    n_bootstrap=10000,  # Number of bootstrap samples
    p_value_delta=0,    # Delta for p-value computation
    random_performance=(0, 0)
)

# Perform contamination tests
results = {}
for file_name in os.listdir(data_dir):
    if mbpp_train_keyword in file_name or humaneval_keyword in file_name:
        # Get model name and benchmark type
        model_name = file_name.split("_")[0]
        benchmark_type = "mbpp_train" if mbpp_train_keyword in file_name else "humaneval"

        # Get the corresponding MBPP test file
        model_file_path = os.path.join(data_dir, file_name)
        mbpp_test_file = match_test_file(file_name, mbpp_test_files)

        # Load MBPP test data
        with open(mbpp_test_file, "r") as f:
            mbpp_test_data = json.load(f)
        mbpp_test_accuracies = calculate_accuracies(mbpp_test_data)

        # Load the current model's data
        with open(model_file_path, "r") as f:
            model_data = json.load(f)
        model_accuracies = calculate_accuracies(model_data)

        # Aggregate distributions
        model_mean_accuracy = np.mean(model_accuracies)
        mbpp_test_mean_accuracy = np.mean(mbpp_test_accuracies)

        # Debugging: Check distribution means
        print("Model mean accuracy:", model_mean_accuracy)
        print("MBPP test mean accuracy:", mbpp_test_mean_accuracy)

        # Generate dummy reference models
        scores_ref_models = np.random.uniform(0, 1, 10)  # Dummy reference models
        scores_ref_models_ref_data = np.random.uniform(0, 1, 10)  # Dummy reference data for comparison

        # Perform contamination test
        result = constat.test(
            np.array([model_mean_accuracy]),       # Model's mean accuracy
            np.array([mbpp_test_mean_accuracy]),   # MBPP test mean accuracy
            scores_ref_models,                     # Dummy reference models
            scores_ref_models_ref_data             # Dummy reference data
        )

        # Store results
        results[file_name] = {
            "model_name": model_name,
            "benchmark": benchmark_type,
            "contamination_results": result
        }
        print(f"Results for {file_name}: {result}")


# Save results to a JSON file
output_path = os.path.join(data_dir, "constat_results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"ConStat results saved to {output_path}")
