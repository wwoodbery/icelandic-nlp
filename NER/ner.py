from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForTokenClassification, DataCollatorWithPadding
from datasets import load_from_disk, load_metric
import os

model_name = "mideind/IceBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name,num_labels=17)
print(model.config.num_labels)
print("LOADING DATASET...")
dataset = load_from_disk("../data/MIM-GOLD-NER/proc_dataset/")
print("DATASET LOADED")

dataset = dataset.train_test_split(test_size=0.3)
print(dataset)
print(dataset["train"][0])

## Print out the first couple rows of the dataset
# print(dataset)
# for i in range(min(5, len(dataset["sentence"]))):  # Change '5' to the number of rows you want to print
#     print("Sentence:", dataset["sentence"][i])
#     print("Labels:", dataset["labels"][i])
#     print("\n")

# Tokenize and encode your dataset
def tokenize_and_encode(examples):
    sentences = [' '.join(tokens) for tokens in examples["sentence"]]
    tokenized = tokenizer(sentences, padding="max_length", truncation=True, max_length=250, return_tensors="pt")
    
    # Make sure the labels are correctly aligned with tokens
    labels = [label + [0] * (250 - len(label)) for label in examples["labels"]]
    
    # Include labels in the tokenized output
    tokenized["labels"] = labels
    
    return tokenized
print("TOKENIZING DATASET...")
encoded_dataset = dataset.map(tokenize_and_encode, batched=True)
print("DATASET TOKENIZED")

print(encoded_dataset["train"][0])

training_args = TrainingArguments(
    output_dir="./training_output/",
    evaluation_strategy="steps",
    eval_steps=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    save_steps=100,
    save_total_limit=2,
    # learning_rate=2e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
)

print("TRAINING MODEL...")
trainer.train()
print("MODEL TRAINED")

results = trainer.evaluate()
print(results)

# Get the predicted labels from the model
predictions = trainer.predict(encoded_dataset["test"])
predicted_labels = predictions.predictions.argmax(axis=2)

# Flatten the true and predicted labels
true_labels = [label for labels in encoded_dataset["test"]["labels"] for label in labels]
predicted_labels = [label for labels in predicted_labels for label in labels]

# Calculate the F1 score
metric = load_metric("seqeval")
results = metric.compute(predictions=predicted_labels, references=true_labels)
f1_score_result = results["overall_f1"]

print("F1 Score:", f1_score_result)