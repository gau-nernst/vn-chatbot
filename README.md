# vn-chatbot

## Set up environment

pip install --upgrade jax jaxlib==0.1.64+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html

For half-precision training with PyTorch

https://github.com/NVIDIA/apex

## Tokenization

Sentencepiece
```
spm_train --input=voz_texts.txt --model_prefix=out --vocab_size=10000 --character_coverage=1.0 --model_type=BPE --max_sentence_length=100000 --num_threads=8
```

## Model

Decoder (look at past tokens to generate next token)

Reversible residual

Bucket input


## design decision

### Tokenization

Hugging Face: better to use their own Tokenizers library

Trax: only support sentencepiece -> use sentence piece

Two options
- byte level
- word level

Two algo
- bpe
- unigram

Other parameters
- vocab size

How i choose
- tokenized sequence length
- speed of training the tokenizer

### Reformer

Axial positional encoding
- No significant improvement
- Cannot have variable sequence length in a batch -> major issue