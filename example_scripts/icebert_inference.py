################################
########### USE PIPE ###########
################################

print('OPTION #1 -- USE A PIPELINE')

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("fill-mask", model="mideind/IceBERT")
print('SOUP MASK: ')
mask = pipe('Súpan var <mask> á bragðið.')
top3_tokens = [x['token_str'] for x in mask][:3]
for token in top3_tokens:
    print('Súpan var {} á bragðið.'.format(token))

print('\n')

################################
########## LOAD MODEL ##########
################################

print('OPTION #2 -- LOAD MODEL DIRECTLY')

# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("mideind/IceBERT")
model = AutoModelForMaskedLM.from_pretrained("mideind/IceBERT")

# Define the masked sentence
masked_sentence = "Súpan var <mask> á bragðið."

# Tokenize the input sentence
inputs = tokenizer(masked_sentence, return_tensors="pt")
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

# Get logits of the masked token
outputs = model(**inputs)
logits = outputs.logits
mask_token_logits = logits[0, mask_token_index, :]

# Get three masked token with highest prob
top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()

for token in top_3_tokens:
    print(masked_sentence.replace(tokenizer.mask_token, tokenizer.decode([token])))