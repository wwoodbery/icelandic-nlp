import os
from tqdm import tqdm

train_path = "/users/wwoodber/data/icelandic-nlp/data/IC3_txt/train"

files = os.listdir(train_path)

with open("/users/wwoodber/data/icelandic-nlp/data/IC3_txt/train.txt", "w", encoding="utf-8") as openfile:
    for file in tqdm(files):
        with open(os.path.join(train_path, file), "r", encoding="utf-8") as f:
            text = f.read()
            openfile.write(text + '\n')
    