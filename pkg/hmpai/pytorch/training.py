from tqdm.notebook import tqdm
from hmpai.pytorch.utilities import DEVICE
import torch


def train(model, train_loader, optimizer, loss_function):
    model.train(True)

    loss_per_batch = []

    for batch in tqdm(train_loader):
        # (Index, samples, channels), (Index, )
        data, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

        optimizer.zero_grad()

        predictions = model(data)

        loss = loss_function(predictions, labels)
        loss_per_batch.append(loss.item())

        loss.backward()

        optimizer.step()

    return loss_per_batch


def validate(model, validation_loader, loss_function):
    model.eval()

    loss_per_batch = []
    total_correct = 0
    total_instances = 0

    with torch.no_grad():
        for batch in tqdm(validation_loader):
            # (Index, samples, channels), (Index, )
            data, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

            predictions = model(data)
            predicted_labels = torch.argmax(predictions, dim=1)
            correct_predictions = sum(predicted_labels == labels).item()

            total_correct += correct_predictions
            total_instances += len(labels)

            loss = loss_function(predictions, labels)
            loss_per_batch.append(loss.item())

    return loss_per_batch, round(total_correct / total_instances, 5)
