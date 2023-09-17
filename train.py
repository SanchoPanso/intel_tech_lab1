import os
import matplotlib.pyplot as plt
import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
import torch.nn.functional as F

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-2
NUM_BATCHES_TO_LOG = 1


def main():
    
    # wandb.login()
    # wandb.init(
    #     project="MNIST",
    # )
    
    # Download training data from open datasets.
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=4)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device, scheduler)
        test(test_dataloader, model, loss_fn, optimizer, device, scheduler)
    
    wandb.finish()
    print("Done!")

    torch.save(model, "model.pth")
    print("Saved PyTorch Model State to model.pth")


def train(dataloader, model, loss_fn, optimizer, device, scheduler):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    
def test(dataloader, model, loss_fn, optimizer, device, scheduler):
    
    # ✨ W&B: Create a Table to store predictions for each test step
    columns=["id", "image", "guess", "truth"]
    for digit in range(10):
      columns.append("score_" + str(digit))
    test_table = wandb.Table(columns=columns)
    log_counter = 0
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            _, predicted = torch.max(pred.data, 1)
            if log_counter < NUM_BATCHES_TO_LOG:
              log_test_predictions(X, y, pred, predicted, test_table, log_counter)
              log_counter += 1
            
    test_loss /= num_batches
    correct /= size
    
    scheduler.step(test_loss)
    lr = optimizer.param_groups[0]["lr"]
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    wandb.log({"acc": correct, "loss": test_loss, "lr": lr})
    wandb.log({"test_predictions" : test_table})
    

    
    


# convenience funtion to log predictions for a batch of test images
def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):
  # obtain confidence scores for all classes
  scores = F.softmax(outputs.data, dim=1)
  log_scores = scores.cpu().numpy()
  log_images = images.cpu().numpy()
  log_labels = labels.cpu().numpy()
  log_preds = predicted.cpu().numpy()
  # adding ids based on the order of the images
  _id = 0
  for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
    # add required info to data table:
    # id, image pixels, model's guess, true label, scores for all classes
    img_id = str(_id) + "_" + str(log_counter)
    test_table.add_data(img_id, wandb.Image(i), p, l, *s)
    _id += 1
    if _id == BATCH_SIZE:
      break

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 3, 3),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 3, 5),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48, 164),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(164, 10),
            nn.Softmax(dim=1),    
        )


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


if __name__ == '__main__':
    main()

