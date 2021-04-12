import os
import tokenizers
from tokenizers import models, normalizers, pre_tokenizers, processors, decoders, trainers
import sentencepiece as spm
from typing import Iterable, List

def get_files(data_dir, full_path=True):
    files = os.listdir(data_dir)
    files = [x for x in files if x.endswith(".txt")]
    if full_path:
        files = [os.path.join(data_dir, x) for x in files]
    return files

def get_threads(files, at_least=1):
    THREAD_TOK = "THREAD"
    POST_TOK = "START_POST"
    
    buffer = []
    post = []
    for x in files:
        with open(x, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                
                # start of a new thread
                if line.startswith(THREAD_TOK):
                    if len(buffer) >= at_least:
                        yield buffer
                    buffer = []
                    
                # start of a new post
                elif line.startswith(POST_TOK):
                    post = " ".join(post)
                    if post: # check for empty post
                        buffer.append(post)
                    post = []
                
                # just text
                else:
                    post.append(line)
                        
    yield buffer

def save_texts(save_path: str="voz_texts.txt", force=False):
    if not os.path.exists(save_path) or force:
        DATA_DIR = "./data"
        files = get_files(DATA_DIR)
        thread_generator = get_threads(files)
        
        # process and write to disk
        with open(save_path, "w", encoding="utf-8") as f:
            for thread in thread_generator:
                for post in thread:
                    f.write(post)
                    f.write("\n")

def train_huggingface_bpe(input_path: str="voz_texts.txt", output: str="voz_tokenizer", vocab_size: int=10000):
    tokenizer = tokenizers.Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFD()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["UNK", "<PAD>", "<BOS>", "<EOS>", "<SEP>"]
    )

    tokenizer.train([input_path], trainer)
    tokenizer.save(f"{output}.json")

def train_sentencepiece_bpe(input_path: str="voz_texts.txt", output: str="voz_out", vocab_size: int=10000):
    spm.SentencePieceTrainer.train(
        input=input_path, 
        model_prefix=output, 
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="BPE",
        max_sentence_length=100000,
        control_symbols=["<sep>"],
        num_threads=8
    )

def main():
    save_texts()
    train_huggingface_bpe(output="voz_tokenizer", vocab_size=15000)
    # train_sentencepiece_bpe(output="voz_out", vocab_size=15000)


if __name__ == "__main__":
    os.environ["RAYON_RS_NUM_CPUS"] = "8"
    main()