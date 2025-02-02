from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

import torch
from torch.utils.data import Dataset

def token_mask(sequence, ignore_token):
    return (sequence == ignore_token).unsqueeze(0).unsqueeze(0)

def positional_mask(sequence):
    return torch.triu(torch.ones((
        1,
        sequence.shape[-1],
        sequence.shape[-1])
    ), diagonal=1).type(torch.bool)

class TokenizedBilingualDataset(Dataset):
    def __init__(
            self,
            dataset,
            input_tokenizer:Tokenizer,
            output_tokenizer:Tokenizer,
            input_lang:str,
            output_lang:str,
            input_sequence_length:int,
            output_sequence_length:int
    ) -> None:
        super().__init__()
        self.ds = dataset
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

    # needed for dataloader
    def __len__(self):
        return len(self.ds)

    # return ids after tokenizer
    def __getitem__(self, idx):
        sos_input_token = torch.tensor(
            [self.input_tokenizer.encode("[SOS]").ids[0]],
            dtype=torch.long
        )
        pad_input_token = (torch.tensor(
            [self.input_tokenizer.encode("[PAD]").ids[0]],
            dtype=torch.long)
        )
        eos_input_token = torch.tensor(
            [self.input_tokenizer.encode("[EOS]").ids[0]],
            dtype=torch.long
        )
        sos_output_token = torch.tensor(
            [self.output_tokenizer.encode("[SOS]").ids[0]],
            dtype=torch.long
        )
        pad_output_token = torch.tensor(
            [self.output_tokenizer.encode("[PAD]").ids[0]],
            dtype=torch.long
        )
        eos_output_token = torch.tensor(
            [self.output_tokenizer.encode("[EOS]").ids[0]],
            dtype=torch.long
        )
        input_sentence = self.ds[idx]['translation'][self.input_lang]
        output_sentence = self.ds[idx]['translation'][self.output_lang]

        input_tokens = self.input_tokenizer.encode(input_sentence).ids
        output_tokens = self.output_tokenizer.encode(output_sentence).ids

        encoder_sequence = torch.cat([
            sos_input_token,
            torch.tensor(input_tokens, dtype=torch.long),
            eos_input_token,
            torch.tensor([pad_input_token for _ in range(
                self.input_sequence_length - len(input_tokens) - 2
            )])
        ])

        decoder_sequence = torch.cat([
            sos_output_token,
            torch.tensor(output_tokens, dtype=torch.long),
            torch.tensor([pad_output_token for _ in range(
                self.output_sequence_length - len(output_tokens) - 1
            )])
        ])

        target_sequence = torch.cat([
            torch.tensor(output_tokens, dtype=torch.long),
            eos_output_token,
            torch.tensor([pad_output_token for _ in range(
                self.output_sequence_length - len(output_tokens) - 1
            )])
        ])

        return {
            "encoder_sequence": encoder_sequence,
            "decoder_sequence": decoder_sequence,
            "target_sequence": target_sequence,
            "encoder_ mask": token_mask(encoder_sequence, pad_input_token),
            "decoder_mask": token_mask(decoder_sequence, pad_output_token)
                            | positional_mask(decoder_sequence),
        }

# approx 3 minutes load time
def get_translation_dataset(input_language:str, output_language:str, input_sequence_length:int, output_sequence_length:int, version:str):
    ds = load_dataset(
        f"{version}",
        f"{input_language}-{output_language}"
    )
    filtered_ds = ds.filter(lambda x: len(x['translation'][input_language]) <= input_sequence_length-2).filter(lambda x: len(x['translation'][output_language]) <= output_sequence_length-2)
    return filtered_ds["train"].train_test_split(test_size=0.1)


def get_translate_iterator(ds, lang):
    for type in ds.keys():
        for item in ds[type]:
            yield item["translation"][lang]

def create_language_tokenizer(ds, lang):
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
    )
    tokenizer.train_from_iterator(
        get_translate_iterator(ds, lang),
        trainer=trainer
    )
    tokenizer.save(f"translator_tokenizer/tokenizer-{lang}.json")
    return tokenizer

def load_tokenizer_from_save(lang, type:str="translator"):
    if type == "translator":
        return Tokenizer.from_file(f"translator_tokenizer/tokenizer-{lang}.json")
    elif type == "language_detection":
        pass
    else:
        print("Unknown type '" + type + "'")
