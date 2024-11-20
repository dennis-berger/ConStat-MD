import json

def count_tests(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Count the number of times "all_tests_passed" is True
    all_tests_passed_count = sum(1 for result in data if result.get("all_tests_passed") == True)
    
    # Count the total number of test cases
    total_tests_count = len(data)
    
    return all_tests_passed_count, total_tests_count

# File path to your JSON file
json_file = "code_generation_results.json"

# Get the counts
all_tests_passed_count, total_tests_count = count_tests(json_file)

print(f"Number of cases where all tests passed: {all_tests_passed_count}")
print(f"Total number of test cases: {total_tests_count}")
