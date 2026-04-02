# GPT-Generated
'''
J. Elizabeth Liebl; jliebl@gmu.edu
03/10/2026
ASR
Project: Replace MultIPA main.py

This code was adapted from https://github.com/ctaguchi/multipa main.py
If you use it, please cite:

@misc{taguchi2023universal,
      title={Universal Automatic Phonetic Transcription into the International Phonetic Alphabet}, 
      author={Chihiro Taguchi and Yusuke Sakai and Parisa Haghani and David Chiang},
      year={2023},
      eprint={2308.03917},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

GPT Generated summary of changes:
Simplified main.py for training a Wav2Vec2-CTC IPA model
using pre-split train/test pandas DataFrames.

This version removes:
    - CommonVoice loading
    - multi-language logic
    - filtering/length limits
    - vocabulary extraction (optional)
    - downsampling
    - TTS/extra sources
'''

# Import Tools
import argparse
import pandas as pd
from datasets import Dataset, Audio, load_from_disk
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)
import torch
import json
from collections import Counter
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


import transformers, sys
print("PYTHON:", sys.executable)
print("TRANSFORMERS VERSION:", transformers.__version__)
print("TRANSFORMERS LOCATION:", transformers.__file__)

# Need this 
def build_vocab_from_dataset(train_ds, test_ds, output_path="/scratch/jliebl/Phonetics/out/vocab.json"):
    """
    Build a vocab.json file from the 'sentence' field of train and test datasets.
    Each unique character becomes a vocabulary entry.

    Args:
        train_ds (Dataset): HuggingFace Dataset for training
        test_ds (Dataset): HuggingFace Dataset for testing
        output_path (str): where to save the vocab.json

    Returns:
        dict: vocabulary dictionary mapping char -> int
    """

    # Collect all transcript strings
    all_text = []

    for example in train_ds:
        all_text.append(example["sentence"])

    for example in test_ds:
        all_text.append(example["sentence"])

    # Count all characters used in the dataset
    counter = Counter()

    for text in all_text:
        counter.update(list(text))

    # Sorted list of unique characters
    vocab_chars = sorted(counter.keys())

    # Build vocabulary dictionary
    vocab_dict = {char: i for i, char in enumerate(vocab_chars)}

    # Add necessary tokens:
    # Wav2Vec2CTCTokenizer requires:
    #   [UNK]  unknown token
    #   [PAD]  padding token
    #   |      word delimiter (instead of space)
    special_tokens = ["[PAD]", "[UNK]", "|"]

    for token in special_tokens:
        if token not in vocab_dict:
            vocab_dict[token] = len(vocab_dict)

    # Save vocab.json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    print(f"Vocabulary saved to {output_path}")
    print(f"Total characters: {len(vocab_dict)}")

    return vocab_dict

# Altered Function to suit
def load_to_dataset(train_ds_path, test_ds_path):
    """Convert pandas DataFrames to HuggingFace Dataset objects."""
    train_ds = load_from_disk(train_ds_path)
    test_ds = load_from_disk(test_ds_path)

    return train_ds, test_ds

# Unaltered: seems fine??
def prepare_dataset(batch, processor):
    """
    Manually load audio files to avoid torchcodec.
    """

    # batch["audio"] is now just a path string
    audio_path = batch["audio"]

    # Load the audio file manually
    audio, sr = sf.read(audio_path)

    # If stereo, convert to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to 16k if needed
    if sr != 16000:
        factor = gcd(sr, 16000)
        up = 16000 // factor
        down = sr // factor
        audio = resample_poly(audio, up, down)

    # Convert raw samples for Wav2Vec2
    batch["input_values"] = processor(
        audio, sampling_rate=16000
    ).input_values[0]

    # Encode IPA transcript
    batch["labels"] = processor(text=batch["sentence"]).input_ids

    return batch

class CustomDataCollatorCTCWithPadding:
    """
    Data collator used for CTC tasks. Dynamically pads the inputs and labels
    using the Wav2Vec2 processor.
    """

    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        # Separate inputs and labels
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Pad inputs
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )

        # Pad labels
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt"
            )

        # Replace padding positions with -100 for CTC loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )

        batch["labels"] = labels
        return batch

def main(args):

    # ---- Load DataFrames ----
    train_ds, test_ds = load_to_dataset(args.train_ds_path, args.test_ds_path)

    if args.create_vocab:
        build_vocab_from_dataset(train_ds, test_ds)
    
    # ---- Load tokenizer + feature extractor ----
    tokenizer = Wav2Vec2CTCTokenizer(
        args.vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )
    
    feature_extractor = Wav2Vec2FeatureExtractor(
        sampling_rate=16000,
        do_normalize=True,
        return_attention_mask=False
    )

    processor = Wav2Vec2Processor(feature_extractor, tokenizer)
    data_collator = CustomDataCollatorCTCWithPadding(processor=processor, padding=True)

    # ---- Prepare data (audio + labels) ----
    train_ds = train_ds.map(lambda x: prepare_dataset(x, processor))
    test_ds = test_ds.map(lambda x: prepare_dataset(x, processor))

    # ---- Load model ----
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name_or_path,
        ctc_loss_reduction="mean",
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer),
        use_safetensors=True
    )

    # ---- Training Arguments ----
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        group_by_length=True,
        remove_unused_columns=False
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        tokenizer=processor   # ← FIXED
    )

    print("TRAINING FOR EPOCHS =", args.epochs)
    print("TRAINING STEPS PER EPOCH =", len(train_ds) // args.batch_size)

    # ---- Train + Save ----
    trainer.train()
    trainer.save_model(args.out_dir)
    processor.save_pretrained(args.out_dir)




# -----------------------------------------------------------
# CLI ARGUMENTS
# -----------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_ds_path", type=str, required=True)
    parser.add_argument("--test_ds_path", type=str, required=True)
    parser.add_argument("--create_vocab", action="store_true")
    parser.add_argument("--vocab_path", type=str, default="/scratch/jliebl/Phonetics/out/vocab.json")
    parser.add_argument("--model_name_or_path", type=str, default="facebook/wav2vec2-large-xlsr-53")
    parser.add_argument("--out_dir", type=str, default="/scratch/jliebl/Phonetics/out")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    main(args)