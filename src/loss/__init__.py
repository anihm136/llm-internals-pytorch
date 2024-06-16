import torch


def estimate_loss(model, eval_dataloader, eval_steps=100):
    model.eval()
    with torch.inference_mode():
        losses = torch.zeros(eval_steps)
        for step in range(eval_steps):
            x, y = eval_dataloader.load_batch()
            _, loss = model(x, y)
            losses[step] = loss.item()

    model.train()

    return losses.mean()
