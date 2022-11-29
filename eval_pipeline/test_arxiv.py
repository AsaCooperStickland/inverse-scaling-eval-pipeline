import json
import matplotlib.pyplot as plt
from collections import defaultdict
import openai
import os
import numpy as np
openai.api_key = os.getenv("OPENAI_API_KEY")

def call_api(model, inputs, max_parallel=20):
    outputs = []

    n_batches = int(np.ceil(len(inputs) / max_parallel))
    for batch_idx in range(n_batches):
        batch_inputs = inputs[
            batch_idx * max_parallel : (batch_idx + 1) * max_parallel
        ]
        batch_outputs = openai.Completion.create(
            model=model,
            prompt=batch_inputs,
            max_tokens=1,
            stop="\n",
            temperature=0,
        )
        for completion in batch_outputs.choices:
            outputs.append(completion.text)
    return outputs
model = "ada:ft-university-of-edinburgh:arxiv-ada-6250-2022-11-23-13-14-17"
test_data_points = 200
load_cache = True
years = [str(i) for i in range(10, 23)]
test_examples = defaultdict(list)
for year in years:
    with open(f"../arxiv_years/{year}.jsonl") as f:
        dataset = f.readlines()
        print(len(dataset))
    for example in dataset:
        test_examples[year].append(json.loads(example))

accuracies = []
for year in years:
    prompts = [example["prompt"] for example in test_examples[year]]
    completions = [example["completion"][:-1] for example in test_examples[year]]
    prompts = prompts[:test_data_points]
    completions = completions[:test_data_points]
    if load_cache:
        predictions = []
        with open(f"../arxiv_years/{year}_completions.jsonl", "r") as f:
            prediction_data = f.readlines()
            for example in prediction_data:
                predictions.append(json.loads(example)["completion"])
    else:
        predictions = call_api(model, prompts)
        with open(f"../arxiv_years/{year}_completions.jsonl", "w") as f:
            for prompt, prediction in zip(prompts, predictions):
                example = {"prompt": prompt, "completion": prediction}
                f.write(json.dumps(example) + "\n")


    correct = 0
    for i, (prediction, target) in enumerate(zip(predictions, completions)):
        # print(prediction)
        # print(target)
        if prediction == target:
            correct += 1
            print(prompts[i])
    accuracy = correct / len(predictions)
    print(f"{year}: {accuracy}")
    accuracies.append(accuracy)

plt.figure()
plt.scatter([2000 + int(year) for year in years], accuracies)
plt.xlabel("Year submitted")
plt.ylabel("Accuracy")
plt.show()
plt.savefig("years.png")

