import torch
from torch.nn import Linear, Sequential, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnext101_32x8d
import torchvision.transforms as T
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_size = 0
val_size = 0

def get_loaders():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = T.Normalize(mean, std)

    train_path = ''
    test_path = ''
    augmentations = T.Compose([
        T.RandomGrayscale(),
        T.RandomHorizontalFlip(),
        T.RandomRotation(30),
        T.RandomVerticalFlip(),
    ])

    train_loader = DataLoader(
        ImageFolder(train_path, transform=T.Compose([
            T.Resize(256),
            augmentations,
            T.ToTensor(),
            normalize
        ])),
        batch_size=4,
        shuffle=True,
    )

    test_loader = DataLoader(
        ImageFolder(test_path, transform=T.Compose([
            T.Resize(256),
            T.ToTensor(),
            normalize
        ])),
        batch_size=4,
        shuffle=True,
    )
    return train_loader, test_loader

def get_accuracy(labels, outputs, inputs):
    ret, predictions = torch.max(outputs.data, 1)
    correct_counts = predictions.eq(labels.data.view_as(predictions))
    acc = torch.mean(correct_counts.type(torch.FloatTensor))
    return acc.item() * inputs.size(0)

def train_fn(loader, model, optimizer, loss_fn, scaler):
    accuracy = 0
    loss = 0
    model.train()
    for _, (data, targets) in enumerate(tqdm(loader)):
        data = data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss += loss.item() * data.size(0)
        accuracy += get_accuracy(targets, scores, data)
    return accuracy, loss

def val_fn(loader, model, loss_fn, scaler):
    accuracy = 0
    loss = 0
    with torch.no_grad():
        model.eval()
        for _, (data, targets) in enumerate(tqdm(loader)):
            data = data.to(device)
            targets = targets.to(device)

            with torch.cuda.amp.autocast():
                scores = model(data)
                loss = loss_fn(scores, targets.float())

            scaler.scale(loss).backward()
            scaler.update()

            loss += loss.item() * data.size(0)
            accuracy += get_accuracy(targets, scores, data)
    return accuracy, loss

def add_to_history(history, train_metrics, val_metrics):
    train_acc, train_loss = train_metrics
    val_acc, val_loss = val_metrics

    avg_train_loss = train_loss/train_size
    avg_train_acc = train_acc/train_size

    avg_valid_loss = val_loss / val_size
    avg_valid_acc = val_acc / val_size

    history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

def main():
    train_loader, test_loader = get_loaders()
    model = resnext101_32x8d(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    model.fc = Linear(2048, 2)

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.1)
    scaler = torch.cuda.amp.GradScaler()
    history = []

    for _ in range(10):
        train_metrics = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        val_metrics = val_fn(test_loader, model, loss_fn, scaler)
        add_to_history(history, train_metrics, val_metrics)

    print(val_fn(test_loader, model, loss_fn, scaler))
