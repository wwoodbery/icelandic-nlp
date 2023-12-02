from datasets import load_dataset, Dataset
import os

def read_conll_file(file_path):
    '''
    Reads in a .txt file in the CoNLL format
    Outputs a list of sentences (list of tokens) and their labels
    '''
    sentences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        current_sentence = []
        current_labels = []
        for line in file:
            line = line.strip()
            if not line:
                if current_sentence:
                    if len(current_sentence) <= 200:
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                current_sentence = []
                current_labels = []
            else:
                try:
                    token, label = line.split('\t')
                    current_sentence.append(token)
                    current_labels.append(label)
                except:
                    pass
    return sentences, labels

# Define the path to your dataset directory
dataset_dir = "../data/MIM-GOLD-NER"

# List of files in the dataset directory
data_files = os.listdir(dataset_dir)
data_files = [x for x in data_files if x not in ['README', 'proc_dataset']]

# Initialize empty lists to store data
all_sentences = []
all_labels = []

# Read data from each file and append to the lists
for data_file in data_files:
    sentences, labels = read_conll_file(os.path.join(dataset_dir, data_file))
    all_sentences.extend(sentences)
    all_labels.extend(labels)
print("ALL FILES READ AND PARSED")

# Create a label-to-id mapping
label_to_id = {}
id_counter = 0
for labels in all_labels:
    for label in labels:
        if label not in label_to_id:
            label_to_id[label] = id_counter
            id_counter += 1

print(id_counter)
# Convert labels to integer IDs
all_label_ids = []
for labels in all_labels:
    label_ids = [label_to_id[label] for label in labels]
    all_label_ids.append(label_ids)
print("LABELS CONVERTED TO INTS")

# Create a dictionary containing your data
data_dict = {
    "sentence": all_sentences,
    "labels": all_label_ids
}

print("CREATING DATASET...")
# Create a Hugging Face dataset
custom_dataset = Dataset.from_dict(data_dict)
print("DATASET CREATED")

print("SAVING DATASET...")
# Save the dataset to a file (e.g., in Arrow format)
custom_dataset.save_to_disk(os.path.join(dataset_dir, 'proc_dataset'))
print("DATSET SAVED")
