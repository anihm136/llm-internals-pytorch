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
            logits, loss = self(idx[:, -8:])
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

        return idx


class AttentionHead(nn.Module):
    def __init__(self, input_dim, head_dim, context_size, mask_previous=True):
        super().__init__()
        self.w_q = nn.Linear(input_dim, head_dim)
        self.w_k = nn.Linear(input_dim, head_dim)
        self.w_v = nn.Linear(input_dim, head_dim)
        if not mask_previous:
            self.register_buffer("mask", torch.ones(context_size, context_size))
        else:
            self.register_buffer(
                "mask", torch.tril(torch.ones((context_size, context_size)))
            )

    def forward(self, idx):
        _, time_steps, data_dim = idx.shape

        key = self.w_k(idx)
        query = self.w_q(idx)
        # Dot-product attention is scaled to prevent convergence
        # of softmax into one-hot vectors
        w = query @ key.transpose(-2, -1) * data_dim**-0.5

        w = w.masked_fill(self.mask[:time_steps, :time_steps] == 0, float("-inf"))
        w = nn.functional.softmax(w, dim=-1)

        value = self.w_v(idx)
        out = w @ value
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self, num_heads, input_dim, head_dim, context_size, mask_previous=True
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                AttentionHead(input_dim, head_dim, context_size, mask_previous)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(num_heads * head_dim, input_dim)

    def forward(self, idx):
        idx = torch.cat([h(idx) for h in self.heads], dim=-1)
        idx = self.proj(idx)
        return idx


class TransformerBlock(nn.Module):
    def __init__(
        self, embedding_dim, num_attention_heads, context_size, scale_inner_ffwd_layer=4
    ):
        super().__init__()
        head_dim = embedding_dim // num_attention_heads
        self.att = MultiHeadAttention(
            num_attention_heads, embedding_dim, head_dim, context_size
        )
        self.ffwd = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * scale_inner_ffwd_layer),
            nn.ReLU(),
            nn.Linear(embedding_dim * scale_inner_ffwd_layer, embedding_dim),
        )

    def forward(self, idx):
        # Attention and feedforward as residual channel operations instead of
        # direct- these should start kicking in later in the optimization process
        idx = idx + self.att(idx)
        idx = idx + self.ffwd(idx)
        return idx


class BigramLanguageModelv2(nn.Module):
    """
    v2 adds transformer blocks for communication and computation between tokens
    """

    def __init__(
        self, vocab_size, context_size, embedding_dim=32, num_attention_heads=4
    ):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embeddings = nn.Embedding(context_size, embedding_dim)
        self.transformer = nn.Sequential(
            TransformerBlock(embedding_dim, num_attention_heads, context_size),
            TransformerBlock(embedding_dim, num_attention_heads, context_size),
            TransformerBlock(embedding_dim, num_attention_heads, context_size),
        )
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        tok_embedding = self.tok_embeddings(idx)
        pos_embedding = self.pos_embeddings(
            torch.arange(idx.shape[1], device=tok_embedding.device)
        )
        embedding = tok_embedding + pos_embedding
        comp = self.transformer(embedding)
        logits = self.lm_head(comp)

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
            logits, loss = self(idx[:, -8:])
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

        return idx
