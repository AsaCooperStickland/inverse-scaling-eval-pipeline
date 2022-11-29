import random
import csv
from datasets import load_dataset

random.seed(22)

def write_bigbench_to_file(dataset_chosen: str) -> None:
    
    dataset = load_dataset("bigbench", dataset_chosen)

    with open(f"../big_bench/{dataset_chosen}.csv", "w") as file:
        
        writer = csv.writer(file)
        writer.writerow(["prompt", "classes", "answer_index"])
        start_index = 1 if dataset_chosen == "strange_stories" else 0 # strange_stories has an extra line of context at the beginning

        for idx, example in enumerate(dataset["validation"]):
            prompt = example["inputs"]
            prompt = prompt.split("\n")
            prompt[start_index] = prompt[start_index].replace("Q", "Select the correct option. Q")
            # print(prompt)
            if dataset_chosen == "strategyqa":
                # strategyqa doesn't give you options in the prompt
                prompt.insert(1, "choice: Yes")
                prompt.insert(1, "choice: No")
                random.shuffle(prompt[start_index + 1:-1]) 
            else:
                random.shuffle(prompt[start_index + 1:-1]) 
                
            for i in range(start_index + 1, len(prompt) - 1):
                prompt[i] = prompt[i].replace("choice:", f"{i - start_index}:")
                prompt[i] = prompt[i].replace("option:", f"{i - start_index}:")
            
            potential_answers = [choice.split(": ")[1] for choice in prompt[start_index + 1:-1]]
            # print(potential_answers)
            # print(example["targets"])
            
            if dataset_chosen == "strategyqa":
                true_answer = example["targets"][0].split(".")[0] # extract yes or no answer
            else:
                true_answer = example["targets"][0]
            
            try:
                answer = potential_answers.index(true_answer)
            except ValueError:
                print(f"Skipping example {idx} because the target isn't in the prompt options")
                continue
            
            options = [f" {i + 1}" for i in range(len(prompt) - 2 - start_index)]
            prompt = "\n".join(prompt)
            print(prompt)
            print(answer)
            scaling_format = [prompt, options, answer]
            writer.writerow(scaling_format)

if __name__ == "__main__":
    dataset_chosen = "strange_stories"
    write_bigbench_to_file(dataset_chosen)
