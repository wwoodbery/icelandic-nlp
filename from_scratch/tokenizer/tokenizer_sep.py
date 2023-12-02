from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path("/users/wwoodber/data/icelandic-nlp/data/leipzig/wiki").glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    '[PAD]',
    '[UNK]',
    '[CLS]',
    '[SEP]',
    '[MASK]',
])

tokenizer.save_model("/users/wwoodber/data/icelandic-nlp/from_scratch/tokenizer")

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

tokenizer = ByteLevelBPETokenizer(
    "/users/wwoodber/data/icelandic-nlp/from_scratch/tokenizer/vocab.json",
    "/users/wwoodber/data/icelandic-nlp/from_scratch/tokenizer/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ("[CLS]", tokenizer.token_to_id("[CLS]")),
)
tokenizer.enable_truncation(max_length=512)

tokenizer.encode("Ég tala smá íslensku.")

tokens = tokenizer.encode("Ég tala smá íslensku.").tokens

print(tokens)