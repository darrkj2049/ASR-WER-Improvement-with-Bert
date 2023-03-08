from datasets import load_dataset
from transformers import AutoTokenizer
import torch

dataset = load_dataset('librispeech_asr', 'clean', split='test' , streaming=True)
Tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

preprocessed_dataset = []



for example in dataset:
    transcription = example['text']
    print(transcription)
#     transcription = example['sentence']
#     tokens = Tokenizer.tokenize(trXanscription)
#     preprocessed_dataset.append({'transcription': tokens})

# Load the pre-trained DistilBert tokenizer
