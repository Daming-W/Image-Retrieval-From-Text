import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import clip_modified

import warnings

warnings.filterwarnings("ignore")


text = ['bird', 'fish', 'spider']
input = 'cute spider'

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def txt_to_vec(text):
    tokens = clip_modified.tokenize(text)
    print(f'tokens: {len(tokens)}')
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([input_ids])

    with torch.no_grad():
        output = model(input_tensor)

    feature_vector = output[0][0].numpy()
    return feature_vector 

text_f = [txt_to_vec(t) for t in text]
input_f = txt_to_vec(input)

print(input_f.shape)