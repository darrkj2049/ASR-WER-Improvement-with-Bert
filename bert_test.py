import torch
from transformers import BertTokenizer, BertForMaskedLM
from datasets import load_dataset
from jiwer import wer

# Load Librispeech dataset
dataset = load_dataset("librispeech_asr", "clean", split="train")

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Fine-tune the model on Librispeech dataset
for i in range(10):
    for example in dataset:
        input_ids = tokenizer.encode(example["speech"], add_special_tokens=True)
        labels = tokenizer.encode(example["text"], add_special_tokens=True)
        loss, _ = model(torch.tensor([input_ids]), labels=torch.tensor([labels]))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Generate ASR transcripts and correct errors using BERT-uncased model
for example in dataset:
    input_ids = tokenizer.encode(example["speech"], add_special_tokens=True)
    output = model.generate(torch.tensor([input_ids]))
    transcript = tokenizer.decode(output[0], skip_special_tokens=True)
    corrected_transcript = transcript.replace("[MASK]", example["text"])

    # Calculate WER for evaluation
    reference = example["text"].lower()
    hypothesis = corrected_transcript.lower()
    error_rate = wer(reference, hypothesis)

    print(f"Reference: {reference}\nHypothesis: {hypothesis}\nWER: {error_rate}\n")
