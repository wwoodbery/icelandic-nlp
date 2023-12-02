from tokenizers import ByteLevelBPETokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import datasets

data_path = "/users/wwoodber/data/icelandic-nlp/data/IC3"

# Load the dataset in .arrow format
dataset = datasets.load_from_disk(data_path)

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Create a list of texts for training the tokenizer
training_texts = []
for example in dataset["train"]:
    training_texts.append(example["text"])

# Customize training
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
tokenizer.train_from_iterator(training_texts, vocab_size=20000, min_frequence = 2, show_progress=True, special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"])

# Save the trained model
tokenizer.save_model("/users/wwoodber/data/icelandic-nlp/from_scratch/tokenizer")

# Load the saved tokenizer
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("/users/wwoodber/data/icelandic-nlp/from_scratch/tokenizer.json")

# Encode text using the loaded tokenizer
encoding = tokenizer.encode("Ég tala smá íslensku.")
tokens = encoding.tokens

print(tokens)