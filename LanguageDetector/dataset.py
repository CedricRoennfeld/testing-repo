from typing import Iterator

import torch
from datasets import get_dataset_config_names, load_dataset
import json

from torch.utils.data import Dataset
import re
from tqdm import tqdm
import random

# Word Explanation:
# Sentence: string consisting of one (or possibly more) sentences
# Token: string containing a single word
# id: unique (in the given contex) number, corresponding to e.g. a token


class WordTokenizer:
    r"""
    Word-Level Tokenizer: provides a class that splits a sentence
    into a list of tokens, encodes these into ids or decodes a list of
    ids to a sentence.

    Attributes:
        unk_token (str): the token that represents an unknown token
        occ (dict[str, int]): dictionary returning the amount of times
            a token occurred in the sentences the tokenizer was
            trained on
        ids (dict(str, int)): dictionary returning id of a sequence
        tokens (dict[int, str]): dictionary returning the token for
            that corresponds to the given id
        token_amount (int): total amount of tokens the tokenizer konws

    Examples:
        >>> Sentences = [
        >>>     "The House is red.",
        >>>     "The Bike is green.",
        >>>     "An Apple is red?",
        >>>     "A House is big?"
        >>> ]
        >>> tokenizer = WordTokenizer.build_from_iterator(
        >>>     (s for s in Sentences),'[UNK]',2
        >>> )
        >>> tokenizer.encode("The House is green")
        [1, 5, 2, 0, 3]
        >>> tokenizer.decode([2, 1, 5, 4, 6])
        "is the house red ?"
    """
    def __init__(
            self,
            occ: dict[str,int] = None,
            unk_token: str = None,
            ids: dict[str,int] = None,
            tokens: dict[int, str] = None,
            token_amount: int = 0
    ) -> None:
        # private constructor for WordTokenizer class
        self.occ = occ if occ else dict()
        self.unk_token = unk_token
        self.ids = ids if ids else dict()
        self.tokens = tokens if tokens else dict()
        self.token_amount = token_amount

    @classmethod
    def build_from_iterator(
            cls,
            sequences_iterator: Iterator[str],
            unk_token: str,
            min_occ: int = 1,
            special_tokens: list[str] = None
    ) -> 'WordTokenizer':
        r"""
        Building and training the tokenizer from multiple sequences

        Parameter:
            sequences_iterator (iterator): an iterator that yields all
                sequences used for training the tokenizer
            unk_token (str): the sequence that represents an unknown
                token
            min_occ (int): amount of times a sequence has to occur for
                the tokenizer to create a token from it []
                Default: 1
            special_tokens list(str): list of special tokens that aren't
                in the sequences the tokenizer is trained on, but should
                still be added (for example: Padding Tokens)
        """

        # initiates empty tokenizer
        word_tokenizer = cls(unk_token=unk_token)
        # add unknown token
        word_tokenizer.ids = {unk_token: 0}
        word_tokenizer.token_amount += 1
        word_tokenizer.occ[unk_token] = 0

        # add special tokens
        for st in special_tokens:
            word_tokenizer.ids[st] = word_tokenizer.token_amount
            word_tokenizer.token_amount += 1
            word_tokenizer.occ[st] = 0

        # create ids dictionary if enough occurrences
        for seq in sequences_iterator:
            for word in WordTokenizer.split(seq):
                # increase occurrences until min_occ is reached for
                # the first time -> add to ids dictionary
                if word not in word_tokenizer.occ:
                    word_tokenizer.occ[word] = 1
                else:
                    word_tokenizer.occ[word] += 1

                if (word not in word_tokenizer.ids
                        and word_tokenizer.occ[word] >= min_occ):
                    word_tokenizer.ids[word] = word_tokenizer.token_amount
                    word_tokenizer.token_amount += 1

        # reverse ids dictionary to get tokens dictionary
        word_tokenizer.tokens = {v: k for k, v in word_tokenizer.ids.items()}

        return word_tokenizer

    @classmethod
    def load_from_save(cls, tokenizer_dict: list[dict]) -> 'WordTokenizer':
        r"""
        Constructor to create a tokenizer from a save list (e.g. json)

        Parameter:
            tokenizer_dict (list[dict]): a dictionary consisting
                of merged occurrences and ids. list should contain
                dicts with the keys "token_str", "id", "occ"
        """
        occ = {entry["token_str"]: entry["occ"] for entry in tokenizer_dict}
        ids = {entry["token_str"]: entry["id"] for entry in tokenizer_dict}
        tokens = {entry["id"]: entry["token_str"] for entry in tokenizer_dict}
        # find first occurrence of id 0
        unk_token = next(
            entry["token_str"] for entry in tokenizer_dict if entry["id"] == 0
        )
        token_amount = len(tokenizer_dict)

        return cls(
            occ=occ,
            unk_token=unk_token,
            ids=ids,
            tokens=tokens,
            token_amount=token_amount
        )

    @staticmethod
    # method for splitting sequences
    def split(seq: str) -> list[str]:
        # Define patterns:
        # pc: punctuation characters of different languages
        # word: \b to start and end a word (e.g. whitespace)
        #       \w+ one or more word character (e.g. letters)
        #       (?:[-']\w+)* optional group of - or ' symbols followed
        #           by at least one word character
        pc = r"[!,;:¡¿.?]"
        word = r"\b\w+(?:[-']\w+)*\b"
        connector = r"[-']"

        # Create a pattern to capture a group that is either
        # a punctuation, a word or a connector
        pattern = f"({pc})|({word})|({connector})"

        # return a list of captured groups
        groups = [match.group() for match in re.finditer(pattern, seq)]
        return groups

    def tokenize(self, seq: str) -> list[dict]:
        # tokenizing a sentence by splitting it and then matching the
        # found token strings (groups) with the corresponding ids
        tokens = list()
        # do not differentiate between "House" and "house"
        # always use lower case
        seq = seq.lower()

        # iterate over token strings
        # if it exists in the dictionary: expand list with it
        # otherwise: add a unknown token
        for token in WordTokenizer.split(seq):
            if token not in self.ids:
                tokens.append({"token": self.unk_token, "id": 0})
            else:
                tokens.append({"token": token, "id": self.ids[token]})
        return tokens

    def encode(self, seq: str) -> list[int]:
        r"""
        encodes a string and returns a list of the found ids
        """
        return [entry['id'] for entry in self.tokenize(seq)]

    def decode(self, tokens: list[int]) -> str:
        r"""
        decodes a list and returns a string of the token strings
        """
        return " ".join(self.tokens[idx] for idx in tokens)


# inherit from torch Dataset to use its DataLoader
class LanguageDetectorTrainer(Dataset):
    r"""
    Trainer for Language Detector, storing training data and tokenizer

    Args:
        min_length (int): minimum length of a sequence
        max_length (int): maximum length of a sequence

    Attributes:
        word_tokenizer (WordTokenizer): the tokenizer for the input
            sequences
        language_tokenizer (dict[str,int]): dictionary that maps a
            language (e.g. "en") to its id
        data (list[dict]): list of pairs consisting of a sentence and
            its language
        languages (set[str]): set of all languages (abreviations)
    """
    def __init__(self, min_length: int, max_length:int ) -> None:
        self.word_tokenizer = None
        self.language_tokenizer = None
        self.min_length = min_length
        self.max_length = max_length
        self.data = list()
        self.languages = set()

    # returns the amount of data
    def __len__(self) -> int:
        return len(self.data)

    # Dataset function to get data entry for Dataloader
    def __getitem__(self, idx: int) -> dict:
        # data entry should consist of the plain sentence and language
        # as well as tbe ids returned by the tokenizer (as tensors)
        seq = self.data[idx]["sequence"]
        lang = self.data[idx]["language"]
        encoded = self.word_tokenizer.encode(seq)
        return {
            "sequence": seq,
            # fill ids with PAD ids to the max length
            "tok_sequence": torch.cat([
                torch.tensor(encoded),
                torch.tensor([
                    self.word_tokenizer.tokenize("[PAD]".lower())[0]['id']
                    for _ in range(self.max_length - len(encoded))
                ]),
            ]),
            "language": lang,
            "tok_language":  torch.tensor(self.language_tokenizer[lang])
        }

    @classmethod
    def build_from_iterator(
            cls,
            min_length: int,
            max_length: int,
            samples_pl: int,
            min_occ: int,
            save_path: str = None
    ) -> 'LanguageDetectorTrainer':
        r"""
        Constructor to build the Trainer from the opus_books dataset

        Parameter:
            min_length (int): minimum length of a sequence
            max_length (int): maximum length of a sequence
            samples_pl (int): samples per language used for training
            min_occ (int): minimum number of occurrences for the
                tokenizer to accept a token
            save_path (str): path to save the trainer
        """
        # create an empty trainer, load dataset and train tokenizer
        language_detector = cls(min_length, max_length)
        language_detector._initiate_dataset(samples_pl)
        language_detector._initiate_tokenizer(min_occ)
        if save_path is not None:
            language_detector.save_to_json(save_path)

        return language_detector

    def _initiate_dataset(self, samples_pl: int) -> None:
        # load a list of all possible configurations
        config_names = get_dataset_config_names("opus_books")
        for config_name in config_names:
            # extract languages in configuration
            lang1, lang2 = config_name.split("-")
            # add languages to set
            self.languages.add(lang1)
            self.languages.add(lang2)

        # empty dictionary to store all sentences of a language
        language_data = {lang: list() for lang in self.languages}

        # loop over every sub dataset and store the sentences in
        # language_data dictionary
        for config_name in config_names:
            lang1, lang2 = config_name.split("-")
            # load sub dataset (first time loading may take longer)
            ds = load_dataset("opus_books", config_name)
            #
            for item in tqdm(
                    ds["train"]["translation"],
                    desc=f"Filter {lang1} -> {lang2}"
            ):
                # add sentences of a language to language_data, only
                # if filter was sucessfull
                for lang in [lang1, lang2]:
                    sequence = item[lang]
                    if self.filter_sentence(sequence):
                        language_data[lang].append({
                            "sequence": sequence.lower(),
                            "language": lang
                        })
        # only use a random 'samples_pl' amount for dataset
        # or everything if amount is less than samples_pl
        for lang in self.languages:
            self.data.extend(random.sample(
                language_data[lang],
                min(samples_pl, len(language_data[lang]))
            ))

    def _initiate_tokenizer(self, min_occ: int) -> None:
        # create word and language tokenizers
        sequence_iterator = (entry["sequence"] for entry in self.data)
        self.word_tokenizer = WordTokenizer.build_from_iterator(
            sequences_iterator=sequence_iterator,
            unk_token="[UNK]",
            min_occ=min_occ,
            special_tokens=["[PAD]"]
        )
        self.language_tokenizer = {
            lang: idx for idx,lang in enumerate(self.languages)
        }



    def filter_sentence(self, sentence: str) -> bool:
        # a sentence is filtered successfully if it only contains a few
        # allowed special tokens and letters
        # clear all allowed special token from sentence
        cleaned_sentence = re.sub(r"[ \!\',;:¡¿\-\.?]", '', sentence)
        # check for sentence length and being alphabetic
        # (after allowed token are removed)
        return (self.min_length <= len(sentence) <= self.max_length
                and cleaned_sentence.isalpha())

    def save_to_json(self, path: str) -> None:
        # save the fully initialized trainer to a json for later usage
        data = {
            "word_tokenizer": [
                {
                    "id": idx,
                    "token_str": token_str,
                    "occ": self.word_tokenizer.occ[token_str]
                } for token_str, idx in self.word_tokenizer.ids.items()
            ],
            "language_tokenizer": [
                {
                    "id": idx,
                    "language": lang
                } for idx,lang in enumerate(self.languages)
            ],
            "sequence_length": {
                "min": self.min_length,
                "max": self.max_length
            },
            "data": [
                {
                    "sequence": entry["sequence"],
                    "language": entry["language"]
                } for entry in self.data
            ]
        }

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    @classmethod
    def load_from_json(cls, path: str) -> 'LanguageDetectorTrainer':
        r"""
        Constructor to load a previous initialized trainer
        from a json file

        Parameter:
            path: path to the data
        """
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            min_length = data["sequence_length"]["min"]
            max_length = data["sequence_length"]["max"]
            language_detector = cls(min_length, max_length)

            language_detector.word_tokenizer = WordTokenizer.load_from_save(
                data["word_tokenizer"]
            )
            language_detector.language_tokenizer = {
                entry["language"]: entry["id"]
                for entry in data["language_tokenizer"]
            }
            language_detector.languages = [
                entry["language"] for entry in data["language_tokenizer"]
            ]
            language_detector.data = data["data"]
            return language_detector
