from torchvision.datasets import CelebA
from torchvision import transforms
import torch
from torch.utils.data import random_split
from torchvision import models
from torch import nn
import argparse
from sklearn.metrics import f1_score, accuracy_score
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
import pickle

class ResNetClassifier(nn.Module):
    def __init__(self, resnet_version="resnet18", num_classes=1): # 1 => binary classifier
        super(ResNetClassifier, self).__init__()

        if resnet_version == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            out_size = 512
        elif resnet_version == "resnet34":
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            out_size = 512
        elif resnet_version == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            out_size = 2048
        elif resnet_version == "resnet101":
            model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            out_size = 2048
        else:
            raise ValueError("Invalid resnet version")

        # drop head
        self.resnet = nn.Sequential(*list(model.children())[:-1])
        # disable gradient
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        self.fc1 = nn.Linear(out_size, out_size)
        self.fc2 = nn.Linear(out_size, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # if called, train a model

    argParser = argparse.ArgumentParser()

    argParser.add_argument("--device", type=str, default="cuda")
    argParser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    argParser.add_argument("--epochs", type=int, default=1) # 1 epoch is enough for fine-tuning
    argParser.add_argument("--lr", type=float, default=1e-3)
    argParser.add_argument("--batch-size", type=int, default=1024)
    argParser.add_argument("--attr-id", type=int, default=2)
    argParser.add_argument("--checkpoint", type=str)

    args = argParser.parse_args()

    ds = CelebA(root='./data', split='all', download=True, transform=transforms.ToTensor())

    argParser = argparse.ArgumentParser()
    argParser.add_argument("--device", type=str, default="cuda")
    argParser.add_argument("--checkpoint", type=str)

    # split `ds` into a train & test -- the test set is then further split into 20 chunks
    # set seed
    # NOTE: we should probably store these sets somewhere
    # (they should be deterministic, but still)
    torch.manual_seed(42)
    test_size = 0.5
    n_chunks = 30
    train_set, test_set = random_split(ds, [1 - test_size, test_size])
    test_sets = random_split(test_set, [1/n_chunks] * n_chunks)

    # attr_id = 2 # attractive
    attr_id = args.attr_id
    # train model
    model = ResNetClassifier(args.backbone).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # bs = 32
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    n_epochs = 1 # fine-tuning epochs
    for epoch in range(args.epochs):
        model.train()
        for x, y in tqdm(train_loader):
            optimizer.zero_grad()
            out = model(x.to(args.device))
            loss = criterion(out, y[:, attr_id].to(args.device).unsqueeze(1).float())
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
            correct = 0
            total = 0
            y_true = []
            y_pred = []
            for x, y in tqdm(test_loader):
                out = model(x.to(args.device))
                y_pred.append((out > 0).cpu().detach().numpy().flatten())
                y_true.append(y[:, attr_id].cpu().detach().numpy().flatten())
            y_true = np.hstack(y_true)
            y_pred = np.hstack(y_pred)
            print(f"Test accuracy: {accuracy_score(y_true, y_pred)}, F1 score (pos): {f1_score(y_true, y_pred)}")
            print()
    

    # save model

    ckpt_dir = "models-ckpt"
    pt_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.pt")
    json_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.json")
    ds_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.pkl")

    torch.save(model, pt_filename)
    # store args in json
    with open(json_filename, "w") as f:
        json.dump(vars(args), f)
    
    with open(ds_filename, "wb") as f:
        pickle.dump({
            "train": train_set,
            "test": test_set,
            "test_chunks": test_sets,
        }, f)