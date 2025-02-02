from pyarrow.dataset import dataset
from torchgen.executorch.api.et_cpp import return_names

from LanguageDetector.train import train


if __name__ == '__main__':
    #train(
    #    min_length=8,
    #    max_length=256,
    #    dataset_sample_size=1000,
    #    transformer_dimension=512,
    #    transformer_ffn_dimension=2048,
    #    encoder_iterations=6,
    #    decoder_iterations=6,
    #    encoder_heads=8,
    #    decoder_heads=8,
    #    dropout=0.0,
    #    optimizer_learning_rate=0.001,
    #    epochs=1000,
    #    current_epoch=100,
    #    model_save_path="LD_model_save/small",
    #    detector_path="LD/small.json",
    #    batch_size=8,
    #)
    train(
        min_length=8,
        max_length=256,
        dataset_sample_size=5000,
        transformer_dimension=512,
        transformer_ffn_dimension=2048,
        encoder_iterations=6,
        decoder_iterations=6,
        encoder_heads=8,
        decoder_heads=8,
        dropout=0.1,
        optimizer_learning_rate=0.001,
        epochs=1000,
        current_epoch=100,
        model_save_path="LD_model_large/ld_large_model",
        detector_path="LD/large_dataset_loader.json",
        batch_size=8,
    )