# GPT-Generated
'''
J. Elizabeth Liebl; jliebl@gmu.edu
03/10/2026
ASR for ComputEL
Generate vocab.json
'''

from datasets import load_from_disk
import json
from collections import Counter


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

models = ["hsb_fam", "hsb_geo", "hsb_phon", "lg_fam", "lg_geo", "lg_phon", "tt_fam", "tt_geo", "tt_phon", "rand"]

for m in models:
    train = load_from_disk(f"/scratch/jliebl/ASR_ComputEL/Datasets/{m}_tr")
    test = load_from_disk(f"/scratch/jliebl/ASR_ComputEL/Datasets/{m}_te")
    vocab = build_vocab_from_dataset(train, test, f"/scratch/jliebl/ASR_ComputEL/ASR_ComputEL/Models/{m}/vocab.json")
    print(f"{m} vocab created!")