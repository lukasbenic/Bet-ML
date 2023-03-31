import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def prepare_data(x_train_path, y_train_path):
    x_train_df = pd.read_csv(x_train_path, index_col=False).drop("Unnamed: 0", axis=1)
    y_train_df = pd.read_csv(y_train_path, index_col=False).drop("Unnamed: 0", axis=1)

    # Do any additional preprocessing here

    x_train_tensor = torch.tensor(x_train_df.values).float()
    y_train_tensor = torch.tensor(y_train_df.values).float()

    return x_train_tensor, y_train_tensor


def train_bayesian_regression(
    svi, x_train, y_train, batch_size, num_epochs, use_cuda=False
):
    x_train_tensor = torch.tensor(x_train).float()
    y_train_tensor = torch.tensor(y_train).float()

    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0
    )

    if use_cuda:
        x_train_tensor = x_train_tensor.cuda()
        y_train_tensor = y_train_tensor.cuda()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for x_batch, y_batch in train_loader:
            if use_cuda:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            # compute loss and gradients
            loss = svi.step(x_batch, y_batch)

            # accumulate epoch loss
            epoch_loss += loss

        # print loss on the same line, with a trailing space to overwrite any leftover characters
        print(
            f"\rTraining loss at epoch {epoch+1}: {epoch_loss/len(train_loader):.4f}",
            end="",
        )
        sys.stdout.flush()

    # print newline character after training is done
    print("\n")
