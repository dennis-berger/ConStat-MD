import json
import numpy as np
# Load the JSON file
with open("code_generation_results.json", "r") as file:
    data = json.load(file)
all_tests_passed_array = np.array([1 if entry["all_tests_passed"] else 0 for entry in data])
print(all_tests_passed_array)
