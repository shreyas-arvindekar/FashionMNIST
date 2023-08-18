from numpy import dtype
import torch
import torch.optim
import torch.utils.data
from torchvision import transforms
from torch.nn.functional import one_hot
from torchvision.datasets import FashionMNIST
from torchmetrics import Accuracy
import matplotlib.pyplot as plt

from model import FashionMNISTModelV0

train_dataset = FashionMNIST(
    root="./", train=True, transform=transforms.ToTensor(), download=True
)

train_data, val_data = torch.utils.data.random_split(
    dataset=train_dataset,
    lengths=[int(0.8 * len(train_dataset)), int(0.2 * len(train_dataset))],
)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

EPOCHS = 10
LR = 0.1
DEVICE = "mps"
mdl = FashionMNISTModelV0().to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.SGD(params=mdl.parameters(), lr=LR)
metric = Accuracy(task="multiclass", num_classes=10).to(DEVICE)

train_accuracies, val_accuracies = [], []
for epoch in range(EPOCHS):
    for idx, (X, y) in enumerate(train_dataloader):
        mdl.train()
        ohe_y = one_hot(y, num_classes=10).type(torch.float).to(DEVICE)
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        # 1 Do a forward pass
        y_logits = mdl(X)
        # 2 Calculate the loss
        loss = loss_fn(y_logits, ohe_y)
        # 3 Optimizer zero grad
        optimizer.zero_grad()
        # 4 Loss backward
        loss.backward()
        # 5 Optimizer step
        optimizer.step()

        if idx % 500 == 0:
            accuracy_score = metric(torch.argmax(y_logits, dim=1), y)
            train_accuracies.append(float(accuracy_score))

            val_pred = None
            yval_true = None
            mdl.eval()
            with torch.inference_mode():
                for xval, yval in val_dataloader:
                    xval, yval = xval.to(DEVICE), yval.to(DEVICE)
                    yval_pred = torch.argmax(mdl(xval), dim=1)
                    if val_pred is None:
                        val_pred = yval_pred
                        yval_true = yval
                    else:
                        val_pred = torch.cat((val_pred, yval_pred))
                        yval_true = torch.cat((yval_true, yval))

            val_accuracy_score = metric(val_pred, yval_true)
            val_accuracies.append(float(val_accuracy_score))
            print(
                f"Epoch: {epoch}\tLoss: {float(loss):.4f}\tAccuracy: {accuracy_score:.2f}\tValidation accuracy score: {val_accuracy_score:.2f}"
            )

torch.save(mdl.state_dict(), "fashionMNIST_TrainedModel.pth")

plt.plot(train_accuracies, marker="o", c="orange", label="Train accuracy", alpha=0.8)
plt.plot(val_accuracies, marker="o", c="green", label="Validation accuracy", alpha=0.8)
plt.xlabel("Batches")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("Model accuracies.png", dpi=1200)
plt.show()
