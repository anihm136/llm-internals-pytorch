class CharTokenizer:
    def __init__(self, text: str):
        self.vocab = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, tokens: list[int]):
        return "".join([self.itos[t] for t in tokens])
