from src.models.bigram_model import (
    BigramLanguageModelv2,
)
from src.tokenizers.char_tokenizer import CharTokenizer
from src.dataloaders.block_loader import BlockLoader
from src.loss import estimate_loss

import torch


if __name__ == "__main__":
    SEED = 1337

    DATA_PATH = "./input.txt"

    BATCH_SIZE = 64
    CONTEXT_LENGTH = 256
    EMBEDDING_DIM = 384
    NUM_ATTENTION_HEADS = 6
    NUM_LAYERS = 6

    TRAIN_STEPS = 5000
    EVAL_INTERVAL_STEPS = 500
    LEARNING_RATE = 3e-4

    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    data = open(DATA_PATH, "r", encoding="utf-8").read()
    tokenizer = CharTokenizer(data)
    data = torch.tensor(tokenizer.encode(data), dtype=torch.long)
    vocab_size = len

    train_size = int(0.9 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    train_loader = BlockLoader(train_data, BATCH_SIZE, CONTEXT_LENGTH, device)
    val_loader = BlockLoader(val_data, BATCH_SIZE, CONTEXT_LENGTH, device)

    # model = BigramLanguageModelv0(tokenizer.vocab_size, tokenizer.vocab_size)
    # model = BigramLanguageModelv1(tokenizer.vocab_size, CONTEXT_LENGTH)
    model = BigramLanguageModelv2(
        tokenizer.vocab_size,
        CONTEXT_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    loss = float("inf")
    for steps in range(TRAIN_STEPS):
        xb, yb = train_loader.load_batch()

        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (steps + 1) % EVAL_INTERVAL_STEPS == 0:
            train_loss = estimate_loss(model, train_loader)
            val_loss = estimate_loss(model, val_loader)
            print(
                "Step: {} | Train loss: {} | Val loss: {}".format(
                    steps + 1, train_loss, val_loss
                )
            )

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokenizer.decode(model.generate(context)[0].tolist()))
