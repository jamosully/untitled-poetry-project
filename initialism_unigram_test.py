from importlib.metadata import distribution
import re
import torch
import torchtext
import numpy as np
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from scipy.stats import multinomial

dataset = load_dataset("albertxu/CrosswordQA")


def format_clues(dataset):
    clues = []
    for sample in dataset:
        if not sample["clue"]:
            continue
        else:
            clue = (
                ["<s>"]
                + word_tokenize(sample["clue"].replace("___", "BLANK"))
                + ["<e>"]
            )
            clues.append(clue)
    print("Clues formatted")
    return clues


def create_vocab(tokenized_dataset):
    return np.unique(np.concatenate(tokenized_dataset)).tolist()


def get_index_for_word(word, vocab):
    if word in vocab:
        index = vocab.index(word)
    else:
        index = -1
    return index


def get_log_probs(data, vocab):
    counts = np.ones((len(vocab), len(vocab)))
    for clue in data:
        for i in range(len(clue) - 1):
            token_1 = clue[i]
            token_2 = clue[i + 1]

            index_1 = vocab.index(token_1)
            index_2 = vocab.index(token_2)
            counts[index_1, index_2] += 1

    first_token_counts = np.sum(counts, axis=1)
    cond_probs = counts / first_token_counts[:, None]
    return np.log(cond_probs)


def initialism_check(clue: str, words: [str]):
    initials = "".join(next(zip(*clue.lower().split())))
    return [a for a in words if a.lower() in initials]


def find_closet_words(word, n):
    initials = dict()
    for l in word:
        if l in initials:
            continue
        else:
            initials[l] = []
    dists = torch.norm(glove.vectors - glove[word], dim=1)
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1])
    for idx, difference in lst[1 : n + 1]:
        if glove.itos[idx][0] in initials:
            initials[glove.itos[idx][0]].append(glove.itos[idx])
    return initials


def generate_unigrams(log_probs, vocab, word, n):
    close_words = find_closet_words(word, n)
    unigrams = []
    for close_word in close_words[word[0]]:
        current_token = close_word
        last_token = ""
        tokens = [close_word]
        while len(tokens) < len(word):
            clue_distribution = np.exp(
                log_probs[get_index_for_word(current_token, vocab), :]
            )
            clue_distribution = clue_distribution / np.sum(clue_distribution)

            sample = multinomial.rvs(1, clue_distribution)
            for l in word.lower():
                if l == word[0]:
                    continue
                for x in range(len(np.where(sample))):
                    for y in range(len(np.where(sample[x]))):
                        xy_sample = np.where(sample)[x][y]
                        print(vocab[xy_sample].lower())
                        sample_word = vocab[xy_sample].lower()
                        if sample_word[0] == l:
                            current_token = sample_word
                            break
                    if current_token != last_token:
                        break
            tokens.append(current_token)
            last_token = current_token
        print(tokens)
        unigrams.append(tokens)
    return unigrams


glove = torchtext.vocab.GloVe(name="6B", dim=50)

train = np.array_split(np.asarray(dataset["validation"]), 7)[1].tolist()
clues = format_clues(train)
vocab = create_vocab(clues)
log_probs = get_log_probs(clues, vocab)
print(generate_unigrams(log_probs, vocab, "mouse", 5))
