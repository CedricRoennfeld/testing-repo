import torch
from torch.utils.data import DataLoader

from transformer import Transformer
from pathlib import Path
from tqdm import tqdm
import warnings
import sys
import os
import matplotlib.pyplot as plt

import Translator.dataset as dataset

def dprint(msg: str, debug_mode: bool):
    if debug_mode:
        print(msg)

# -> (1, seq_length)
def translate(
        model,
        input_sequence,
        encoder_mask,
        output_tokenizer,
        output_sequence_length
):

    device = next(model.parameters()).device
    sos_token = output_tokenizer.encode("[SOS]").ids[0]
    eos_token = output_tokenizer.encode("[EOS]").ids[0]
    pad_token = output_tokenizer.encode("[PAD]").ids[0]

    encoder_output = model.encode(input_sequence, encoder_mask)
    # decoder_sequence start with SOS Token and rest is PAD
    decoder_sequence = torch.empty(1,output_sequence_length).fill_(pad_token).int().to(device)
    decoder_sequence[0,0] = sos_token

    pos = 1

    while True:
        # max length reached
        if pos == output_sequence_length:
            break

        tm = dataset.token_mask(decoder_sequence, pad_token)
        pm = dataset.positional_mask(decoder_sequence).to(device)

        decoder_mask = tm | pm
        decoder_output = model.decode(decoder_sequence, encoder_output, decoder_mask, encoder_mask)

        # (B, seq_length, dimension) -> (B, seq_length, vocab_size)
        prediction_distribution = model.predict(decoder_output)

        print( torch.argmax(prediction_distribution[0], dim=-1))

        # view(1) because it returns a zero-dimensional vector
        predicted_token = torch.argmax(prediction_distribution[0][pos]).view(1)
        decoder_sequence[0,pos] = predicted_token
        #decoder_sequence = torch.cat((decoder_sequence, predicted_token), dim=-1)

        if predicted_token == eos_token:
            break
        pos += 1
    #print(decoder_sequence)
    return decoder_sequence

def validation(
        model: torch.nn.Module,
        ds,
        input_tokenizer,
        output_tokenizer,
        output_sequence_length,
        limit
):
    model.eval()
    device = next(model.parameters()).device

    count = 0
    with torch.no_grad():
        for batch in ds:
            #print(batch)
            # (1,seq_length)
            prediction = translate(
                model,
                batch["encoder_sequence"].to(device),
                batch["encoder_mask"].to(device),
                output_tokenizer,
                output_sequence_length
            )
            print("-"*12)
            print(f"input     : {input_tokenizer.decode(batch["encoder_sequence"].tolist()[0])}")
            print(f"target    : {output_tokenizer.decode(batch["target_sequence"].tolist()[0])}")
            print(f"prediction: {output_tokenizer.decode(prediction.tolist()[0])}")

            count += 1

            if count == limit:
                print("-"*12)
                break


def train(
        input_language: str,
        output_language: str,
        version: str,
        batch_size: int,
        input_sequence_length: int,
        output_sequence_length: int,
        transformer_dimension: int,
        transformer_ffn_dimension: int,
        encoder_iterations: int,
        decoder_iterations: int,
        encoder_heads: int,
        decoder_heads: int,
        dropout: float,
        optimizer_learning_rate: float,
        epochs: int,
        model_save_path: str = None,
        current_epoch: int = 0,
        debug_mode: bool = False,
):

    if not debug_mode:
        warnings.filterwarnings("ignore")

    dprint("Loading data...",debug_mode)
    ds = dataset.get_translation_dataset(
        input_language,
        output_language,
        input_sequence_length,
        output_sequence_length,
        version
    )
    dprint("Data loaded successfully.", debug_mode)

    input_tokenizer_path = f"translator_tokenizer/tokenizer-{input_language}.json"
    if Path(input_tokenizer_path).exists():
        input_tokenizer = dataset.load_tokenizer_from_save(input_language)
        dprint(f"Loaded existing tokenizer for {input_language}.", debug_mode)
    else:
        dprint(f"Creating tokenizer for {input_language}...", debug_mode)
        input_tokenizer = dataset.create_language_tokenizer(ds, input_language)
        dprint(f"Tokenizer created successfully.", debug_mode)
    output_tokenizer_path = f"translator_tokenizer/tokenizer-{output_language}.json"
    if Path(output_tokenizer_path).exists():
        output_tokenizer = dataset.load_tokenizer_from_save(output_language)
        dprint(f"Loaded existing tokenizer for {output_language}.", debug_mode)
    else:
        dprint(f"Creating tokenizer for {input_language}...", debug_mode)
        output_tokenizer = dataset.create_language_tokenizer(ds, output_language)
        dprint(f"Tokenizer created successfully.", debug_mode)

    tokenized_train_dataset = dataset.TokenizedBilingualDataset(
        ds["train"],
        input_tokenizer,
        output_tokenizer,
        input_language,
        output_language,
        input_sequence_length,
        output_sequence_length
    )
    tokenized_validation_dataset = dataset.TokenizedBilingualDataset(
        ds["test"],
        input_tokenizer,
        output_tokenizer,
        input_language,
        output_language,
        input_sequence_length,
        output_sequence_length
    )

    train_dataloader = DataLoader(
        tokenized_train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available.")
        print(f"Training on {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print("  Total Memory: "
                  f"{torch.cuda.get_device_properties(i).total_memory 
                     / 1e9} GB")
    else:
        device = torch.device("cpu")
        print("CUDA is not available.")
        print("Training on CPU.")
        print("Notice: Its recommended to train on GPU using Cuda "
              "when working with large models. Consider using CUDA "
              "if available for your GPU.")

    dprint("Initializing model...", debug_mode)
    model = Transformer(
        input_tokenizer.get_vocab_size(),
        output_tokenizer.get_vocab_size(),
        input_sequence_length,
        output_sequence_length,
        transformer_dimension,
        encoder_iterations,
        decoder_iterations,
        encoder_heads,
        decoder_heads,
        dropout,
        transformer_ffn_dimension
    )
    model.to(device)
    dprint("Model initialized successfully.", debug_mode)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_learning_rate
    )

    loss_function = torch.nn.CrossEntropyLoss(
        ignore_index=output_tokenizer.encode("[PAD]").ids[0],
        label_smoothing=0.1
    ).to(device)

    #writer = SummaryWriter("TestWriter")

    if Path(f"{model_save_path}-{current_epoch}").exists():
        print(f"Preload from {model_save_path}-{current_epoch}.")
        state_dict = torch.load(f"{model_save_path}-{current_epoch}")
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        total_batches_run = state_dict["total_batches_run"]
        loss_values = state_dict["loss_values"]
    else:
        print("No preload available, start training from scratch.")
        total_batches_run = 0
        loss_values = []

    while current_epoch < epochs:
        current_epoch += 1
        print(f"current epoch: {current_epoch}, "
              f"total batches: {total_batches_run}",
              flush=True)
        torch.cuda.empty_cache()
        model.train()

        batches = tqdm(train_dataloader, file=sys.stdout)

        epoch_loss = 0
        epoch_batches_run = 0
        for batch in batches:
            # amount of data may not be divisible by batch_size,
            # resulting in the last batch having smaller size => skip
            if batch["encoder_sequence"].shape[0] < batch_size:
                break
            encoder_sequence = batch["encoder_sequence"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            encoder_output = model.encode(encoder_sequence, encoder_mask)
            decoder_output = model.decode(
                batch["decoder_sequence"].to(device),
                encoder_output,
                batch["decoder_mask"].to(device),
                batch["encoder_mask"].to(device)
            )
            prediction_output = model.predict(decoder_output)

            loss = loss_function(
                prediction_output.view(
                    batch_size*output_sequence_length,
                    output_tokenizer.get_vocab_size()
                ),
                batch["target_sequence"].to(device).view(
                    batch_size*output_sequence_length
                )
            )
            epoch_loss += loss.item()
            loss.backward()

            # Log the loss
            #writer.add_scalar('train loss', loss.item(), total_batches_run)
            #writer.flush()

            optimizer.step()
            optimizer.zero_grad()

            epoch_batches_run += 1

        loss_values.append(epoch_loss/epoch_batches_run)

        total_batches_run += epoch_batches_run

        if model_save_path:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "total_batches_run": total_batches_run,
                "loss_values":loss_values
            }, f"{model_save_path}-{current_epoch}")

        # shuffle each epoch again
        validation_dataloader = DataLoader(
            tokenized_validation_dataset,
            batch_size=1,
            shuffle=True
        )
        validation(model, validation_dataloader, input_tokenizer, output_tokenizer, output_sequence_length, 10)
        #print(loss_values)

    print("---")
    print("Training finished.")
    print([epoch+1 for epoch in range(epochs)])
    print(loss_values)

    plt.plot([epoch+1 for epoch in range(epochs)], loss_values)
    plt.show()


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"

    train("de", "en", "opus_books", 8, 256, 256, 512, 2048, 6, 6, 8, 8, 0.1, 1e-4, 500, "translator_model_save/opus-books-de-en",100, debug_mode=False)


