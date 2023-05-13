import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from torch.utils.data import TensorDataset, DataLoader
import pyro.nn as pnn
from sklearn.preprocessing import StandardScaler
import pyro.infer


def prepare_data(x_train_path: str, y_train_path: str):
    scaler = StandardScaler()
    x_train_df = pd.read_csv(x_train_path)
    y_train_df = pd.read_csv(y_train_path)

    clm = x_train_df.columns
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train_df), columns=clm)

    x_train_tensor = torch.tensor(x_train_scaled.values).float()
    y_train_tensor = torch.tensor(y_train_df.values.ravel()).float()
    return x_train_tensor, y_train_tensor, scaler, clm


def train_bayesian_regression(
    svi, x_train_tensor, y_train_tensor, batch_size, num_epochs, device
):
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss = svi.step(x_batch, y_batch)

            normalized_loss = loss / x_batch.shape[0]

            losses.append(normalized_loss)
        if epoch % 50 == 0:
            print(f"iter: {epoch}, normalized loss:{round(normalized_loss,2)}")

    trained_model = svi.guide
    # Convert the list of losses to a pandas DataFrame
    losses_df = pd.DataFrame(losses, columns=["Normalized Loss"])
    # Save the DataFrame as a CSV file
    losses_df.to_csv("models/metrics/losses.csv", index_label="Epoch")

    return trained_model


def predict(model, guide, x_test_tensor, num_samples):
    predictive = pyro.infer.Predictive(
        model, guide=guide, num_samples=num_samples, return_sites=("obs",)
    )
    samples = predictive(x_test_tensor)
    predictions = samples["obs"].mean(dim=0)
    return predictions
