import json
from random import shuffle
import random
from collections import Counter, defaultdict
random.seed(42)

with open("/Users/asa/Downloads/arxiv-metadata-oai-snapshot.json") as f:
    dataset = f.readlines()[::10]
    print(len(dataset))
    shuffle(dataset)

year_counter = Counter()
year_store = defaultdict(list)

for example in dataset:
    data = json.loads(example)
    # print(data["id"])
    year_id = data["id"]
    if "/" in year_id:
        year = year_id.split("/")[1][:2]
    else:
        year = year_id[:2]
    year_counter[year] += 1
    # print(f"{year} {year_counter[year]}")
    if year_counter[year] <= 2000:
        year_store[year].append({"prompt": data["abstract"] + "\n\n###\n\n", "completion": " " + year + "\n"})
    else:
        if year_counter[year] == 2001:
            print(f"finished {year}")
        continue

print(year_counter)
training_data = []

for year in year_store:
    if year_counter[year] > 2000:
        training_data.extend(year_store[year][1000:1250])
        with open(f"../arxiv_years/{year}.jsonl", "w") as file:
            for example in year_store[year][:1000]:
                file.write(json.dumps(example) + "\n")

print(len(training_data))

with open("../arxiv_years/train.jsonl", "w") as file:
    for example in training_data:
        file.write(json.dumps(example) + "\n")