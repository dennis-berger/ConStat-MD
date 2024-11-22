from openai import OpenAI

client = OpenAI()
import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from tqdm import tqdm


def rephrase_prompt(prompt, system_prompt):
    """
    Rephrases a single prompt using OpenAI API.
    Args:
        prompt (str): The original prompt.
        system_prompt (str): The instruction prompt for rephrasing.
    Returns:
        str: The rephrased prompt.
    """
    try:
        response = client.chat.completions.create(model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"### Original Prompt\n{prompt}"}
        ],
        max_tokens=200,
        temperature=0.7)
        rephrased_prompt = response.choices[0].message.content
        # Extract the rephrased part after ### Rephrased Prompt
        if "### Rephrased Prompt" in rephrased_prompt:
            rephrased_prompt = rephrased_prompt.split("### Rephrased Prompt")[-1].strip()
        return rephrased_prompt
    except Exception as e:
        print(f"Error rephrasing prompt: {e}")
        return prompt  # Return the original prompt if rephrasing fails

def rephrase_dataset(dataset, system_prompt):
    """
    Rephrases the `prompt` field of the dataset.
    Args:
        dataset (Dataset): The original dataset.
        system_prompt (str): The instruction prompt for rephrasing.
    Returns:
        Dataset: The updated dataset with rephrased prompts.
    """
    rephrased_data = {key: [] for key in dataset.features.keys()}  # Initialize an empty dictionary for columns

    for row in tqdm(dataset):
        rephrased_row = row.copy()
        rephrased_row["prompt"] = rephrase_prompt(row["prompt"], system_prompt)  # Rephrase the prompt

        for key, value in rephrased_row.items():
            rephrased_data[key].append(value)  # Append each value to the corresponding column list

    return Dataset.from_dict(rephrased_data)


if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")

    # System prompt for rephrasing
    system_prompt = """Significantly rephrase the given prompt while preserving its meaning. Use different vocabulary and sentence structure.

Format your reply as:
### Rephrased Prompt
[Rephrased prompt]
"""

    # Rephrase the dataset
    rephrased_dataset = rephrase_dataset(dataset, system_prompt)

    # Save the rephrased dataset if needed
    print(rephrased_dataset)
    rephrased_dataset.save_to_disk("rephrased_dataset")
