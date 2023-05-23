import os
import pandas as pd
import numpy as np
import argparse
import torch
# import classla
import nltk.data
from tqdm import tqdm
from logzero import logger

from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForNextSentencePrediction


def sliding_window_coherence_evaluation(model, tokenizer, device, sentences, sent_index, max_context_len):
    """ Function is intended to be able to measure the coherence of a story longer than the maximum length of the model"""

    # Tokenize the story
    indexed_tokens = [tokenizer.encode(sent) for sent in sentences]

    # Create a sliding window of max context len on which we will evaluate the model
    context_before, context_after = extract_story_context(indexed_tokens, sent_index, max_context_len)
    context_after_len = len(context_after)

    leftEdgeSum = 0
    rightEdgeSum = 0
    minIndex = sent_index
    maxIndex = sent_index

    for i in range(sent_index, 0, -1):
        if leftEdgeSum + len(indexed_tokens[i]) < max_context_len:
            leftEdgeSum += len(indexed_tokens[i])
            minIndex -= 1
        else:
            break

    for i in range(sent_index, len(indexed_tokens)):
        if rightEdgeSum + leftEdgeSum + len(indexed_tokens[i]) < max_context_len:
            rightEdgeSum += len(indexed_tokens[i])
            maxIndex += 1
        else:
            break

    minIndex = max(minIndex, 0)
    maxIndex = min(maxIndex, len(indexed_tokens) - 1)


    all_loss_w_sent = []
    all_loss_wo_sent = []

    # Compute the probability of the sentence given the context
    for i in range(minIndex, maxIndex + 1):
        if sent_index < i:
            break
        context_before, context_after = extract_story_window(indexed_tokens, i, sent_index, max_context_len)
        input_tensor_w_sent = torch.tensor(context_before + indexed_tokens[sent_index] + context_after).unsqueeze(0).to(device)
        input_tensor_wo_sent = torch.tensor(context_before + context_after).unsqueeze(0).to(device)

        with torch.no_grad():
            # print(input_tensor_wo_sent.shape)
            # print(input_tensor_w_sent.shape)
            outputs_w_sent = model(input_tensor_w_sent)
            outputs_wo_sent = model(input_tensor_wo_sent)

            logits_w = outputs_w_sent.logits
            logits_wo = outputs_wo_sent.logits

            # Shift so that tokens < n predict n
            shift_logits_w = logits_w[..., :-1, :].contiguous()
            shift_labels_w = input_tensor_w_sent[..., 1:].contiguous()

            shift_logits_wo = logits_wo[..., :-1, :].contiguous()
            shift_labels_wo = input_tensor_wo_sent[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss_w_sent = loss_fct(shift_logits_w.view(-1, shift_logits_w.size(-1)), shift_labels_w.view(-1))
            loss_wo_sent = loss_fct(shift_logits_wo.view(-1, shift_logits_wo.size(-1)), shift_labels_wo.view(-1))

        all_loss_w_sent.append(loss_w_sent[-context_after_len:].sum().item() / context_after_len)
        all_loss_wo_sent.append(loss_wo_sent[-context_after_len:].sum().item() / context_after_len)

    return np.mean(all_loss_w_sent), np.mean(all_loss_wo_sent)


def extract_story_context(indexed_tokens, sent_index, max_context_len):
    len_list = [len(tokenized_sent) for tokenized_sent in indexed_tokens]
    target_sent_len = len_list[sent_index]

    full_context_before_len = sum(len_list[:sent_index])
    full_context_after_len = sum(len_list[sent_index + 1:])

    tmp_context_before_len = (max_context_len - target_sent_len) * 0.5
    tmp_context_after_len = (max_context_len - target_sent_len) * 0.5

    if sum(len_list) <= max_context_len:  # full_context_before_len <= context_before_len and full_context_after_len <= context_after_len
        max_context_before_len, max_context_after_len = full_context_before_len, full_context_after_len
    elif full_context_before_len <= tmp_context_before_len and full_context_after_len > tmp_context_after_len:
        max_context_before_len = full_context_before_len
        max_context_after_len = max_context_len - (full_context_before_len + target_sent_len)
    elif full_context_before_len > tmp_context_before_len and full_context_after_len <= tmp_context_after_len:
        max_context_after_len = full_context_after_len
        max_context_before_len = max_context_len - (full_context_after_len + target_sent_len)
    elif full_context_before_len > tmp_context_before_len and full_context_after_len > tmp_context_after_len:
        max_context_before_len = tmp_context_before_len
        max_context_after_len = tmp_context_after_len
    else:
        print("you wrote wrong if statement")
        exit(-1)

    # extract context before
    context_before = []
    context_before_len = []
    for tokenized_sent, sent_len in zip(indexed_tokens[:sent_index - 1][::-1], len_list[:sent_index - 1][::-1]):
        if sum(context_before_len) + sent_len <= max_context_before_len:
            context_before = context_before + tokenized_sent[::-1]
            context_before_len.append(sent_len)

    # extract context after
    context_after = []
    context_after_len = []
    for tokenized_sent, sent_len in zip(indexed_tokens[sent_index:], len_list[sent_index:]):
        if sum(context_after_len) + sent_len <= max_context_after_len:
            context_after = context_after + tokenized_sent
            context_after_len.append(sent_len)

    context_before.reverse()

    return context_before, context_after


def extract_story_window(indexed_tokens, left_edge, sent_index, max_context_len):
    len_list = [len(tokenized_sent) for tokenized_sent in indexed_tokens]
    context_before = []
    context_after = []

    for i in range(sent_index, left_edge - 1, -1):
        if len(context_before) + len_list[i] < max_context_len - len_list[sent_index]:
            context_before += indexed_tokens[i]
        else:
            break

    for i in range(sent_index, len(indexed_tokens)):
        if len(context_after) + len(context_before) + len_list[i] < max_context_len - len_list[sent_index]:
            context_after += indexed_tokens[i]
        else:
            break

    return context_before, context_after


def compute_rest_of_story_probability(model, device, indexed_sentences, sent_index, max_context_len):
    context_before, context_after = extract_story_context(indexed_sentences, sent_index, max_context_len)
    context_after_len = len(context_after)

    input_tensor_w_sentence = torch.tensor(context_before + indexed_sentences[sent_index] + context_after).unsqueeze(0).to(device)
    input_tensor_wo_sentence = torch.tensor(context_before + context_after).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs_w_sentence = model(input_tensor_w_sentence)
        outputs_wo_sentence = model(input_tensor_wo_sentence)

        logits_w = outputs_w_sentence.logits
        logits_wo = outputs_wo_sentence.logits

        # Shift so that tokens < n predict n
        shift_logits_w = logits_w[..., :-1, :].contiguous()
        shift_labels_w = input_tensor_w_sentence[..., 1:].contiguous()

        shift_logits_wo = logits_wo[..., :-1, :].contiguous()
        shift_labels_wo = input_tensor_wo_sentence[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        loss_w_sentence = loss_fct(shift_logits_w.view(-1, shift_logits_w.size(-1)), shift_labels_w.view(-1))
        loss_wo_sentence = loss_fct(shift_logits_wo.view(-1, shift_logits_wo.size(-1)), shift_labels_wo.view(-1))

    normalized_loss_w_sentence = loss_w_sentence[-context_after_len:].sum().item() / context_after_len
    normalized_loss_wo_sentence = loss_wo_sentence[-context_after_len:].sum().item() / context_after_len

    return normalized_loss_w_sentence, normalized_loss_wo_sentence


def compute_story_probability_with_t5(model, tokenizer, device, indexed_sentences, sent_index):
    context_before, context_after = extract_story_context(indexed_sentences, sent_index, 20000)

    # Add instruction to the beginning of the context
    instruction = tokenizer('summarize:')['input_ids']
    context_before = instruction + context_before

    input_tensor_w_sentence = torch.tensor(context_before + indexed_sentences[sent_index] + context_after).unsqueeze(0).to(device)
    input_tensor_wo_sentence = torch.tensor(context_before + context_after).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs_w_sentence = model(input_tensor_w_sentence, labels=input_tensor_w_sentence)
        outputs_wo_sentence = model(input_tensor_wo_sentence, labels=input_tensor_wo_sentence)

        logits_w = outputs_w_sentence.logits
        logits_wo = outputs_wo_sentence.logits

        # Shift so that tokens < n predict n
        shift_logits_w = logits_w[..., :-1, :].contiguous()
        shift_labels_w = input_tensor_w_sentence[..., 1:].contiguous()

        shift_logits_wo = logits_wo[..., :-1, :].contiguous()
        shift_labels_wo = input_tensor_wo_sentence[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        loss_w_sentence = loss_fct(shift_logits_w.view(-1, shift_logits_w.size(-1)), shift_labels_w.view(-1))
        loss_wo_sentence = loss_fct(shift_logits_wo.view(-1, shift_logits_wo.size(-1)), shift_labels_wo.view(-1))

        return loss_w_sentence.item(), loss_wo_sentence.item()

def compute_story_probability_with_bert(model, device, indexed_sentences, sent_index):
    # We will peform next sentence prediction from the context before to the context after with and without the sentence
    context_before, _ = extract_story_context(indexed_sentences, sent_index, 512)
    context_after = indexed_sentences[sent_index + 1]

    # We will say that regardless of the inclusion of the sentence, the sentence is the next sentence
    labels = torch.LongTensor([1]).to(device)

    input_tensor_w_sentence = torch.tensor(context_before + indexed_sentences[sent_index]).unsqueeze(0).to(device)
    input_tensor_wo_sentence = torch.tensor(context_before + context_after).unsqueeze(0).to(device)

    # We will peform next sentence prediction from the context before
    with torch.no_grad():
        outputs_w_sentence = model(input_tensor_w_sentence, labels=labels)
        outputs_wo_sentence = model(input_tensor_wo_sentence, labels=labels)

        # We will take the probability of the next sentence being the sentence  (index 1)
        loss_w_sentence = outputs_w_sentence.logits[0][1].item()
        loss_wo_sentence = outputs_wo_sentence.logits[0][1].item()

        return loss_w_sentence, loss_wo_sentence

def estimate_coherence_with_model(args):

    logger.info("Loading data...")
    df = pd.read_csv("../data/data.csv", index_col=0)

    # Load the text data for each story

    logger.info(f"Loading model {args.model}...")

    try:
        if 'gpt' in args.model or 'bert' in args.model and not args.nsp:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            model = AutoModelForCausalLM.from_pretrained(args.model)
        elif 'bert' in args.model and args.nsp:
            tokenizer = BertTokenizer.from_pretrained(args.model)
            model = BertForNextSentencePrediction.from_pretrained(args.model)
        elif 't5' in args.model:
            tokenizer = T5Tokenizer.from_pretrained(args.model)
            model = T5ForConditionalGeneration.from_pretrained(args.model)
        else:
            logger.error(f"Model {args.model} not supported")
            exit(1)
    except Exception as e:
        logger.error(f"Error loading model {args.model}: {e}")
        exit(1)

    if args.gpu >= 0:
        logger.info("Using GPU")
        device = torch.device('cuda')
        model.to(device)
    else:
        device = torch.device('cpu')
    model.eval()

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.eval()
    max_content_len = args.contextlen

    logger.info("Estimating coherence...")
    progressbar = tqdm(total=df.shape[0], desc="Iteration over folktales")

    for i, row in df.iterrows():
        # Load the sentences data
        if args.language == "en":
            title = row['eng_title'].replace(" ", "_")
        else:
            title = row['slo_title'].replace(" ", "_")

        if not os.path.isfile(f"../data/labeled/{title}.csv"):
            continue

        logger.info(title)
        df_story = pd.read_csv(f"../data/labeled/{title}.csv")
        sentences = df_story['sentence'].tolist()
        labels = df_story['label'].tolist()

        indexed_sentences = [tokenizer.encode(sent) for sent in sentences]
        progressbar2 = tqdm(total=len(sentences), desc="Iteration over sentences")

        results = []
        with_sentence = 0
        without_sentence = 0
        for j, sentence in enumerate(indexed_sentences[:-1]):
            # If last sentence, we use the special token <|endoftext|>
            if 'gpt' in args.model or 'bert' in args.model and not args.nsp:
                if j == len(sentences) - 2:
                    if args.use_sliding_window:
                        with_sentence, without_sentence = sliding_window_coherence_evaluation(model, device, indexed_sentences, j, max_content_len)
                    else:
                        with_sentence, without_sentence = compute_rest_of_story_probability(model, device, indexed_sentences, j, max_content_len)
                else:
                    if args.use_sliding_window:
                        with_sentence, without_sentence = sliding_window_coherence_evaluation(model, device, indexed_sentences[:-1], j, max_content_len)
                    else:
                        with_sentence, without_sentence = compute_rest_of_story_probability(model, device, indexed_sentences[:-1], j, max_content_len)
            elif 't5' in args.model:
                with_sentence, without_sentence = compute_story_probability_with_t5(model, tokenizer, device, indexed_sentences, j)
            elif 'bert' in args.model and args.nsp:
                with_sentence, without_sentence = compute_story_probability_with_bert(model, device, indexed_sentences, j)

            results.append([sentences[j], with_sentence, without_sentence, without_sentence - with_sentence, labels[j]])

            progressbar2.update(1)

        results_df = pd.DataFrame(results, columns=["sentence", "with_sentence", "without_sentence", "difference", "label"])
        if args.use_sliding_window:
            title = f"{title}_window"
        if args.nsp:
            title = f"{title}_nsp"
        results_df.to_csv(f"{args.output}/{title}_{args.model.replace('/', '_')}.csv", sep=",")
        progressbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1, help="use gpu or not")
    parser.add_argument("--model", type=str, default="cjvt/gpt-sl-base", help="Model name")
    parser.add_argument("--language", type=str, default="slo", help="Language (slo or eng)")
    parser.add_argument("--contextlen", type=int, default=1024, help="Context length")
    parser.add_argument("--use_sliding_window", type=bool, default=False, help="Use sliding window")
    parser.add_argument("--nsp", type=bool, default=False, help="Use next sentence prediction")
    parser.add_argument("--input", type=str, default="../data/data.csv", help="Path to the data")
    parser.add_argument("--output", type=str, default="../results", help="Path to the output")

    args = parser.parse_args()

    estimate_coherence_with_model(args)
