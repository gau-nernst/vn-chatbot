import os
import random
from tokenizer import get_files, get_threads

import numpy as np
import torch
from transformers import (
    PreTrainedTokenizerFast,
    ReformerConfig,
    ReformerModelWithLMHead,
    Trainer, 
    TrainingArguments
)

from typing import List

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_convo(threads: List[List[str]]):
    POST_TOKEN = "<SEP>"
    
    for thread in threads:
        result = [""]
        for post in thread:
            result.append(post)
        
        yield POST_TOKEN.join(result)

def get_all_convo():
    files = get_files("./data")
    threads = get_threads(files)
    convo_generator = get_convo(threads)
    
    all_convo = [x for x in convo_generator]
    random.shuffle(all_convo)

    return all_convo

def get_tokenizer(path: str="voz_tokenizer.json"):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=path,
        unk_token="<UNK>",
        pad_token="<PAD>",
        bos_token="<BOS>",
        eos_token="<EOS>",
    )
    return tokenizer

class VozDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=float("inf")):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        encodings = self.tokenizer(
            self.texts[i], 
            return_token_type_ids=False
        )
        if len(encodings.input_ids) > self.max_length:
            return encodings.input_ids[:self.max_length]
        else:
            return encodings.input_ids

class VozCollator():
    def __init__(self, align_by, pad_value=0):
        self.align_by = align_by
        self.pad_value = pad_value
    
    def __call__(self, batch):
        length = max([len(x) for x in batch])
        length = np.ceil(length/self.align_by).astype(int) * self.align_by   # align to chunk size
        
        attention_mask = [[1]*len(x) + [0]*(length-len(x)) if len(x) < length else [1]*length for x in batch]
        input_ids = [x + [self.pad_value]*(length-len(x)) if len(x) < length else x[:length] for x in batch]

        output = {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }
        
        return output

def compute_metrics(pred):
    non_padded_indices = (pred.label_ids != -100)
    
    labels = pred.label_ids[..., 1:][non_padded_indices[..., 1:]]
    pred = np.argmax(pred.predictions[:, :-1], axis=-1)[non_padded_indices[..., :-1]]
    acc = np.mean(np.asarray(pred == labels), dtype=np.float)

    return {"accuracy": acc}

def main():
    all_convo = get_all_convo()
    tokenizer = get_tokenizer()
    VOCAB_SIZE = tokenizer.vocab_size
    sequence_length = 2**14

    train_size = int(len(all_convo)*0.95)
    train_convo = all_convo[:train_size]
    test_convo = all_convo[train_size:]

    train_dataset = VozDataset(train_convo, tokenizer, sequence_length)
    test_dataset = VozDataset(test_convo, tokenizer, sequence_length)

    collator = VozCollator(64, tokenizer.pad_token_id)

    config = ReformerConfig(
        axial_pos_embds=False,
        num_buckets=32,
        vocab_size=VOCAB_SIZE,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        is_decoder=True,
        max_position_embeddings=sequence_length,
        num_attention_heads=2,
    )

    training_args = TrainingArguments(
        output_dir="./",
        num_train_epochs=5,
        do_train=True,
        do_eval=True,
        
        fp16=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        prediction_loss_only=True,
        
        learning_rate=1e-3,
        warmup_steps=1000,
        weight_decay=0.01,

        evaluation_strategy="steps",
        eval_steps=50,
        
        save_steps=1000,
        save_total_limit=10,
        
        # logging_dir='./logs',
        logging_steps=50,
    )

    model = ReformerModelWithLMHead(config)
    model.train()

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    trainer.save_model("voz_model")

if __name__=="__main__":
    main()