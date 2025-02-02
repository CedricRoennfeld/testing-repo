import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from LanguageDetector.dataset import LanguageDetectorTrainer
from pathlib import Path

from transformer import Transformer
import matplotlib.pyplot as plt

def train(
        min_length: int,
        max_length: int,
        dataset_sample_size: int,
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
        min_occ: int=1,
        batch_size: int=1,
        detector_path: str = None,


):

    # Dataset + Tokenizer
    if detector_path is None or not Path(detector_path).exists():
        LD = LanguageDetectorTrainer.build_from_iterator(min_length, max_length, dataset_sample_size, min_occ, detector_path)
    else:
        LD = LanguageDetectorTrainer.load_from_json(detector_path)


    dataloader = DataLoader(LD, batch_size=batch_size, shuffle=True)

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

    model = Transformer(
        len(LD.word_tokenizer.ids),
        len(LD.language_tokenizer),
        max_length,
        1,
        transformer_dimension,
        encoder_iterations,
        decoder_iterations,
        encoder_heads,
        decoder_heads,
        dropout,
        transformer_ffn_dimension
    )
    model.to(device)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_learning_rate
    )

    pad_idx = LD.word_tokenizer.tokenize("[PAD]".lower())[0]['id']

    loss_function = torch.nn.CrossEntropyLoss(
        label_smoothing=0.1
    ).to(device)

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
        torch.cuda.empty_cache()
        model.train()

        # batch shape: {'sequence': str, 'language': str}

        batches = tqdm(dataloader, desc=f"Prcessing Epoch {current_epoch}")

        epoch_loss = 0
        epoch_batches_run = 0

        for batch in batches:
            # amount of data may not be divisible by batch_size,
            # resulting in the last batch having smaller size => skip
            if batch["tok_sequence"].shape[0] < batch_size:
                break

            encoder_sequence = batch["tok_sequence"].to(device)
            encoder_mask = (encoder_sequence == pad_idx).view(batch_size, 1, max_length).to(device)


            encoder_output = model.encode(encoder_sequence, encoder_mask)
            decoder_output = model.decode(
                torch.zeros(batch_size,1,device=device, dtype=torch.long),
                encoder_output,
                encoder_mask=encoder_mask
            )

            prediction_output = model.predict(decoder_output)

            loss = loss_function(prediction_output.squeeze(1), batch["tok_language"].to(device))
            #if math.isnan(loss.item()):
            #print("prediction: ", prediction_output.squeeze(1), "\nlang:", batch["tok_language"].to(device))
            if torch.isnan(loss):
                continue
            epoch_loss += loss.item()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_batches_run += 1

        print(epoch_loss / epoch_batches_run)
        loss_values.append(epoch_loss / epoch_batches_run)

        total_batches_run += epoch_batches_run
        if current_epoch % 10 == 0:
            if model_save_path:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "total_batches_run": total_batches_run,
                    "loss_values": loss_values
                }, f"{model_save_path}-{current_epoch}")

    print("---")
    print("Training finished.")
    print([epoch + 1 for epoch in range(epochs)])
    print(loss_values)

    plt.plot([epoch + 1 for epoch in range(epochs)], loss_values)
    plt.show()