#!/usr/bin/env python3
import ipdb  # noqa: F401
import random  # noqa: F401
import numpy as np  # noqa: F401
import torch
import math
from argparse import ArgumentParser  # noqa: F401
from transformers_per import AutoModelWithLMHead, AutoTokenizer
from statistics import mean


def main(args):
    ratios = []
    for i in range(5):
        ratios.append(run_generation(args))
    mean_ratio = mean(ratios)
    print('Penalty factor: {} - Without penalty / penalty ratio avg: {}'.format(args.rep_pen, millify(mean_ratio)))


def run_generation(args):
    model = AutoModelWithLMHead.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_input_words = (
        torch.tensor(tokenizer.encode(args.input)).unsqueeze(0) if args.input else None
    )
    return model.generate(
        tokenized_input_words,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_ids=tokenizer.eos_token_id,
        repetition_penalty=args.rep_pen,
        max_length=100,
    )
#    generated_words = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
#    print(generated_words)


def millify(n):
    n = float(n)
    millidx = max(0, min(21, int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
    return '{:.0f}e{}'.format(n / 10**(3 * millidx), 3 * millidx)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilgpt2")
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--rep_pen", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
