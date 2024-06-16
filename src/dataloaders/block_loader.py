import torch


class BlockLoader:
    def __init__(
        self,
        data: torch.LongTensor,
        batch_size: int,
        block_size: int,
        device: str = "cpu",
    ):
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

    def load_batch(self):
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([self.data[i : i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1 : i + self.block_size + 1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
