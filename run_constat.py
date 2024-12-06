import os
import json
import numpy as np
from constat import ConStat

# Paths
data_dir = "../processed_results/"

# Initialize ConStat
constat = ConStat(
    n_bootstrap=10000,  # Number of bootstrap samples
    p_value_delta=0,    # Delta for p-value computation
    random_performance=(0, 0)  # Adjust based on use case
)

# Helper function to calculate accuracies from JSON
def calculate_accuracies(json_data):
    return np.array([1 if entry["all_tests_passed"] else 0 for entry in json_data])

# Gather benchmark files
benchmarks = {
    "mbpp_test": "google-research-datasets_mbpp_test_results.json",
    "mbpp_train": "google-research-datasets_mbpp_train_results.json",
    "humaneval": "openai_humaneval_test_results.json",
}

# Load benchmark data
benchmark_accuracies = {}
for benchmark_name, benchmark_file in benchmarks.items():
    benchmark_path = os.path.join(data_dir, benchmark_file)
    with open(benchmark_path, "r") as f:
        benchmark_accuracies[benchmark_name] = calculate_accuracies(json.load(f))

# Initialize storage for results
model_accuracies = {}
results = {}

# Loop through all JSON files in the directory
for file_name in os.listdir(data_dir):
    if file_name.endswith(".json"):
        # Determine the benchmark type
        if "mbpp_test_results" in file_name:
            benchmark_key = "mbpp_test"
        elif "mbpp_train_results" in file_name:
            benchmark_key = "mbpp_train"
        elif "humaneval_test_results" in file_name:
            benchmark_key = "humaneval"
        else:
            continue  # Skip non-test files

        # Load model JSON
        with open(os.path.join(data_dir, file_name), "r") as f:
            model_data = json.load(f)

        # Extract accuracies
        accuracies_model = calculate_accuracies(model_data)

        # Extract model name from filename
        model_name = file_name.split("_")[0]

        # Generate dummy reference model accuracies (replace with real data if available)
        reference_model_count = 10  # Assume 10 reference models
        reference_accuracies = np.random.randint(0, 2, (reference_model_count, len(benchmark_accuracies[benchmark_key])))

        # Run contamination test with MBPP test as the primary reference
        result = constat.test(
            accuracies_model,
            benchmark_accuracies["mbpp_test"],  # Always use MBPP test results as reference
            reference_accuracies,
            benchmark_accuracies[benchmark_key]  # Compare against train or other benchmarks
        )

        # Store the results
        results[file_name] = {
            "model_name": model_name,
            "benchmark_name": benchmark_key,
            "result": result,
        }

# Output results
for file_name, result_data in results.items():
    print(f"Results for {file_name}:")
    print(json.dumps(result_data, indent=4))

# Save results to a JSON file
output_path = os.path.join(data_dir, "constat_results_with_train.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_path}")
