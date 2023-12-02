import torch
var = torch.cuda.is_available()
print(var)
if not var:
    quit()


from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("/users/wwoodber/data/icelandic-nlp/from_scratch/tokenizer", max_len=512)

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)

print(model.num_parameters())
# => 84 million parameters

from transformers import LineByLineTextDataset
from pathlib import Path
from tqdm import tqdm

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/users/wwoodber/data/icelandic-nlp/data/IC3_txt/train.txt",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="/users/wwoodber/data/icelandic-nlp/from_scratch/model/IC3",
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

print("TRAINING MODEL...")
trainer.train()
print("MODEL TRAINED")

print("SAVING MODEL...")
trainer.save_model("/users/wwoodber/data/icelandic-nlp/from_scratch/model")
print("MODEL SAVED")