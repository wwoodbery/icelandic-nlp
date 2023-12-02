from transformers import pipeline, RobertaTokenizerFast

fill_mask = pipeline(
    "fill-mask",
    model="/users/wwoodber/data/icelandic-nlp/from_scratch/model/leipzig_10",
    tokenizer=RobertaTokenizerFast.from_pretrained("/users/wwoodber/data/icelandic-nlp/from_scratch/tokenizer", max_len=512)
)

print("Þetta er bróðir <mask>.")
print(fill_mask("Þetta er bróðir <mask>."))

print('\n')

print("Supán <mask> mjög góð.")
print(fill_mask("Supán <mask> mjög góð."))