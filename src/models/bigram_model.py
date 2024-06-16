import torch.nn as nn
import torch


class BigramLanguageModelv0(nn.Module):
    """
    v0 has a single linear embedding layer that is directly used as logits
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.embeddings(idx)

        if targets is not None:
            batch_size, time_step, data_dim = logits.shape
            reshaped_logits = logits.view(batch_size * time_step, data_dim)
            targets = targets.view(batch_size * time_step)
            loss = nn.functional.cross_entropy(reshaped_logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens=100):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

        return idx


class BigramLanguageModelv1(nn.Module):
    """
    v1 separates the embedding layer from the LM layer, and also adds positional embeddings to add information about token positions
    """

    def __init__(self, vocab_size, context_size, embedding_dim=32):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embeddings = nn.Embedding(context_size, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        tok_embedding = self.tok_embeddings(idx)
        pos_embedding = self.pos_embeddings(
            torch.arange(idx.shape[1], device=tok_embedding.device)
        )
        embedding = tok_embedding + pos_embedding
        logits = self.lm_head(embedding)

        if targets is not None:
            batch_size, time_step, data_dim = logits.shape
            reshaped_logits = logits.view(batch_size * time_step, data_dim)
            targets = targets.view(batch_size * time_step)
            loss = nn.functional.cross_entropy(reshaped_logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens=100):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

        return idx
