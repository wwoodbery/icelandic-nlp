from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

data_path = "/users/wwoodber/data/icelandic-nlp/data/IC3_txt/train"

paths = [str(x) for x in Path(data_path).glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

print("Tokenizer Training...")
# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
print("Tokenizer Trained")

tokenizer.save_model("/users/wwoodber/data/icelandic-nlp/from_scratch/tokenizer")

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

tokenizer = ByteLevelBPETokenizer(
    "/users/wwoodber/data/icelandic-nlp/from_scratch/tokenizer/vocab.json",
    "/users/wwoodber/data/icelandic-nlp/from_scratch/tokenizer/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

tokenizer.encode("Ég tala smá íslensku.")

tokens = tokenizer.encode("Ég tala smá íslensku.").tokens

print(tokens)