from transformers import AutoTokenizer
import random
import string
import jsonlines
import numpy as np
from tqdm import tqdm
import argparse


def generate_text(seq_len, ratio, tokenizer):
    sentence_list = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence makes our life easier.",
        "Learning Python is a valuable skill in the tech industry today.",
        "The sun rises in the east and sets in the west.",
        "Rome wasn't built in a day.",
        "All that glitters is not gold.",
        "The pen is mightier than the sword.",
        "A journey of a thousand miles begins with a single step.",
        "The early bird catches the worm.",
        "Honesty is the best policy.",
    ]

    # Generate a random secret key
    # secret_key = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    secret_key = "".join(random.choices(string.digits, k=5))

    # Determine the position of the key sentence based on the ratio
    key_position = int(seq_len * ratio)

    # Generate the haystack text
    text = ""
    total_tokens = 0
    while total_tokens < seq_len or key_position < 1e14:
        if total_tokens > key_position:
            key_position = 1e15
            key_sentence = f"The secret key is {secret_key}. Remember it. "
            text += key_sentence
            total_tokens += len(tokenizer.tokenize(key_sentence))
        else:
            sentence = random.choice(sentence_list)
            text += sentence + " "
            total_tokens += len(tokenizer.tokenize(sentence))

    text += "Honesty is the best policy.\n\nWhat is the secret key?\nThe Key is: "

    return text, secret_key, total_tokens, ratio


def main():
    parser = argparse.ArgumentParser(description="Generate offline needle-in-a-haystack JSONL data.")
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_path", default="benchmark/datasets/needle_in_haystack_tasks.jsonl")
    parser.add_argument("--seq_lens", default="16000,32000,64000,128000,256000,512000")
    parser.add_argument("--samples_per_setting", type=int, default=5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    seq_lens = [int(item) for item in args.seq_lens.split(",") if item.strip()]
    ratios = list(np.arange(0.01, 0.92, 0.1)) + [0.99]

    tasks = []

    for seq_len in seq_lens:
        for ratio in tqdm(ratios, desc=f"{seq_len}"):
            for i in range(args.samples_per_setting):
                task_text, secret_key, total_tokens, ratio = generate_text(
                    seq_len, ratio, tokenizer
                )
                tasks.append(
                    {
                        "task": task_text,
                        "answer": secret_key,
                        "total_tokens": total_tokens,
                        "ratio": ratio,
                    }
                )

    with jsonlines.open(args.output_path, mode="w") as writer:
        writer.write_all(tasks)


if __name__ == "__main__":
    random.seed(42)
    main()
