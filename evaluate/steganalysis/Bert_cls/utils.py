import torch
import torch.nn.functional as F


def train(model, dataloader, optimizer):
    model.train()
    for batch in dataloader:
        device = model.device

        labels = batch["label"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask=attention_mask)

        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()


def test(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            device = model.device
            labels = batch["label"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == labels).sum().item()
            total += len(labels)
    return correct / total
