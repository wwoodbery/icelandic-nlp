import datasets
from pathlib import Path

# Load the dataset in .arrow format
dataset = datasets.load_dataset("mideind/icelandic-common-crawl-corpus-IC3")

# Specify the output directory for .txt files
train_output_directory = Path("/users/wwoodber/data/icelandic-nlp/data/IC3_txt/train")
test_output_directory = Path("/users/wwoodber/data/icelandic-nlp/data/IC3_txt/test")
validation_output_directory = Path("/users/wwoodber/data/icelandic-nlp/data/IC3_txt/validation")

# Iterate through the dataset and save each example to a .txt file

print("Start train set")
i = 0
for example in dataset["train"]:
    text = example["text"]
    filename = f"train_{i}.txt"
    i += 1
    with open(train_output_directory / filename, "w", encoding="utf-8") as file:
        file.write(text)
print("End train set")

print("Start test set")
i = 0
for example in dataset["test"]:
    text = example["text"]
    filename = f"test_{i}.txt"
    i += 1
    with open(test_output_directory / filename, "w", encoding="utf-8") as file:
        file.write(text)
print("End test set")

print("Start validation set")
i = 0
for example in dataset["validation"]:
    text = example["text"]
    filename = f"validation_{i}.txt"
    i += 1
    with open(validation_output_directory / filename, "w", encoding="utf-8") as file:
        file.write(text)
print("End validation set")