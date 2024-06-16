from src.models.bigram_model import BigramLanguageModelv0
from src.tokenizers.char_tokenizer import CharTokenizer
from src.dataloaders.block_loader import BlockLoader
from src.loss import estimate_loss

import torch


if __name__ == "__main__":
    SEED = 1337

    DATA_PATH = "./input.txt"

    BATCH_SIZE = 32
    CONTEXT_LENGTH = 8

    TRAIN_STEPS = 10000
    EVAL_INTERVAL_STEPS = 500

    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = open(DATA_PATH, "r", encoding="utf-8").read()
    tokenizer = CharTokenizer(data)
    data = torch.tensor(tokenizer.encode(data), dtype=torch.long)
    vocab_size = len

    train_size = int(0.9 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    train_loader = BlockLoader(train_data, BATCH_SIZE, CONTEXT_LENGTH, device)
    val_loader = BlockLoader(val_data, BATCH_SIZE, CONTEXT_LENGTH, device)

    model = BigramLanguageModelv0(tokenizer.vocab_size)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss = float("inf")
    for steps in range(TRAIN_STEPS):
        xb, yb = train_loader.load_batch()

        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if steps + 1 % EVAL_INTERVAL_STEPS == 0:
            train_loss = estimate_loss(model, train_loader)
            val_loss = estimate_loss(model, val_loader)
            print(
                "Step: {} | Train loss: {} | Val loss: {}".format(
                    steps + 1, train_loss, val_loss
                )
            )

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokenizer.decode(model.generate(context)[0].tolist()))
