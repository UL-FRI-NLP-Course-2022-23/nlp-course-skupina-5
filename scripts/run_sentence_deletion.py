import os
import pandas as pd
import argparse
import torch
import classla
import nltk.data
from tqdm import tqdm
from logzero import logger

from transformers import GPT2Tokenizer, GPT2LMHeadModel


def extract_story_context(indexed_tokens, sent_index, max_context_len):
    len_list = [len(tokenized_sent) for tokenized_sent in indexed_tokens]
    target_sent_len = len_list[sent_index]

    context_before = []
    context_after = []

    for tokenized_sent in indexed_tokens[:sent_index]:
        context_before += tokenized_sent

    for tokenized_sent in indexed_tokens[sent_index + 1:]:
        context_after += tokenized_sent

    edge = max(max_context_len - target_sent_len, max_context_len)

    return context_before[:edge], context_after[:edge]


def compute_rest_of_story_probability(model, device, indexed_tokens, sent_index, max_context_len):
    context_before, context_after = extract_story_context(indexed_tokens, sent_index, max_context_len)
    context_after_len = len(context_after)

    input_tensor_w_sentence = torch.tensor(context_before + indexed_tokens[sent_index] + context_after).unsqueeze(0).to(device)
    input_tensor_wo_sentence = torch.tensor(context_before + context_after).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs_w_sentence = model(input_tensor_w_sentence)
        outputs_wo_sentence = model(input_tensor_wo_sentence)

        logits_w = outputs_w_sentence.logits
        logits_wo = outputs_wo_sentence.logits

        # Shift so that tokens < n predict n
        shift_logits_w = logits_w.contiguous()
        shift_logits_wo = logits_wo.contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        loss_w_sentence = loss_fct(shift_logits_w.view(-1, shift_logits_w.size(-1)), input_tensor_w_sentence.view(-1))
        loss_wo_sentence = loss_fct(shift_logits_wo.view(-1, shift_logits_wo.size(-1)), input_tensor_wo_sentence.view(-1))

    if context_after_len == 0:
        return 100, 100
    unnormalized_loss_w_sentence = loss_w_sentence[-context_after_len:].sum().item() / context_after_len
    unnormalized_loss_wo_sentence = loss_wo_sentence[-context_after_len:].sum().item() / context_after_len

    return unnormalized_loss_w_sentence, unnormalized_loss_wo_sentence


def estimate_coherence(args):
    logger.info("Loading data...")
    df = pd.read_csv("../data/data.csv", index_col=0)

    logger.info(f"Loading model {args.model}...")

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model)

    if args.gpu >= 0:
        logger.info("Using GPU")
        device = torch.device('cuda')
        model.to(device)
    else:
        device = torch.device('cpu')

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.eval()
    max_content_len = args.contextlen

    logger.info("Estimating coherence...")
    progressbar = tqdm(total=df.shape[0], desc="Iteration over folktales")

    nltk.download('punkt')
    for i, row in df.iterrows():
        if row['eng_title'] == "THE STRAW, THE COAL, AND THE BEAN":
            if args.language == "slo":
                text = row["slo_text"]
                sentences = nltk.sent_tokenize(text, language="slovene")
            else:
                text = row["eng_text"]
                sentences = nltk.sent_tokenize(text, language="english")
            indexed_sentences = [tokenizer.encode(sentence) for sentence in sentences]

            progressbar2 = tqdm(total=len(indexed_sentences), desc="Iteration over sentences")

            results = []
            for j, sentence in enumerate(indexed_sentences):
                with_sentence, without_sentence = compute_rest_of_story_probability(model, device, indexed_sentences, j, max_content_len)

                results.append([sentences[j], with_sentence, without_sentence, without_sentence - with_sentence])

                progressbar2.update(1)

            results_df = pd.DataFrame(results, columns=["sentence", "with_sentence", "without_sentence", "difference"])
            results_df.to_csv(f"{args.output}/data_{row['slo_title']}.csv", sep=",")
        progressbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1, help="use gpu or not")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--language", type=str, default="en", help="Language (slo or eng)")
    parser.add_argument("--contextlen", type=int, default=1024, help="Context length")
    parser.add_argument("--input", type=str, default="../data/data.csv", help="Path to the data")
    parser.add_argument("--output", type=str, default="../results", help="Path to the output")

    estimate_coherence(parser.parse_args())
