import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
import pyro.contrib.autoguide as autoguide
from torch.utils.data import TensorDataset, DataLoader
import pyro.nn as pnn


class RegressionModel(pnn.PyroModule):
    def __init__(self, input_features, output_features=1):
        super().__init__()
        self.linear = torch.nn.Linear(input_features, output_features)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x)).squeeze(-1)


class BayesianRegressionModel:
    def __init__(self, num_features):
        self.regression_model = RegressionModel(num_features)
        self.num_features = num_features

    def model(self, x_data, y_data):
        # Define the priors for the weight and bias parameters
        weight_prior = dist.Normal(
            torch.zeros(self.num_features, 1), torch.ones(self.num_features, 1)
        ).to_event(2)
        bias_prior = dist.Normal(torch.tensor([0.0]), torch.tensor([10.0]))

        # Wrap the priors using Pyro's plate construct
        priors = {"linear.weight": weight_prior, "linear.bias": bias_prior}
        lifted_module = pyro.random_module("module", self.regression_model, priors)

        # Instantiate the lifted module (i.e., sample the priors)
        lifted_regression_model = lifted_module()

        with pyro.plate("data", len(x_data)):
            prediction_mean = lifted_regression_model(x_data).squeeze()
            pyro.sample(
                "obs",
                dist.Normal(prediction_mean, 1.0),
                obs=y_data.squeeze(1),
            )

    def guide(self, x_data, y_data):
        weight_loc = pyro.param("weight_loc", torch.randn(self.num_features, 1))
        weight_scale = pyro.param(
            "weight_scale",
            torch.ones(self.num_features, 1),
            constraint=dist.constraints.positive,
        )
        bias_loc = pyro.param("bias_loc", torch.randn(1))
        bias_scale = pyro.param(
            "bias_scale", torch.tensor(1.0), constraint=dist.constraints.positive
        )

        weight_dist = dist.Normal(weight_loc, weight_scale).to_event(2)
        bias_dist = dist.Normal(bias_loc, bias_scale)

        # Wrap the priors using Pyro's plate construct
        dists = {"linear.weight": weight_dist, "linear.bias": bias_dist}
        lifted_module = pyro.random_module("module", self.regression_model, dists)

        # Instantiate the lifted module (i.e., sample the priors)
        lifted_regression_model = lifted_module()

        with pyro.plate("data", len(x_data)):
            prediction_mean = lifted_regression_model(x_data).squeeze()
            pyro.sample("obs", dist.Normal(prediction_mean, 1.0))


def prepare_data(x_train_path, y_train_path):
    x_train_df = pd.read_csv(x_train_path).drop(columns=["Unnamed: 0"])
    y_train_df = pd.read_csv(y_train_path).drop(columns=["Unnamed: 0"])

    x_train_tensor = torch.tensor(x_train_df.values).float()
    y_train_tensor = torch.tensor(y_train_df.values).float()
    # y_train_tensor = y_train_tensor.unsqueeze(
    #     -1
    # )  # Add an extra dimension to match the output shape
    return x_train_tensor, y_train_tensor


def train_bayesian_regression(
    svi, x_train_tensor, y_train_tensor, batch_size, num_epochs
):
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(num_epochs):
        # Compute the ELBO loss for this mini-batch

        loss = svi.step(x_train_tensor, y_train_tensor)

        normalized_loss = loss / x_train_tensor.shape[0]

        # Tabulate the loss for plotting
        losses.append(normalized_loss)
        if epoch % 250 == 0:
            print(f"iter: {epoch}, normalized loss:{round(normalized_loss,2)}")

    # Return the trained model
    trained_model = svi.guide
    return trained_model


def model_gamma2(X, y):
    pyro.enable_validation(True)

    min_value = torch.finfo(X.dtype).eps
    max_value = torch.finfo(X.dtype).max

    # We still need to calculate our linear combination

    intercept_prior = dist.Normal(0.0, 1.0)
    linear_combination = pyro.sample(f"beta_intercept", intercept_prior)

    # Also define coefficient priors
    for i in range(X.shape[1]):
        coefficient_prior = dist.Normal(0.0, 1.0)
        beta_coef = pyro.sample(f"beta_{[i]}", coefficient_prior)

        linear_combination = linear_combination + (X[:, i] * beta_coef)

    # But now our mean will be e^{linear combination}
    mean = torch.exp(linear_combination).clamp(min=min_value, max=max_value)

    # We will also define a rate parameter
    rate = pyro.sample("rate", dist.HalfNormal(scale=10.0)).clamp(min=min_value)

    # Since mean = shape/rate, then the shape = mean * rate
    shape = mean * rate

    # Now that we have the shape and rate parameters for the
    # Gamma distribution, we can draw samples from it and condition
    # them on our observations

    with pyro.plate("data", y.shape[0]):
        outcome_dist = dist.Gamma(shape.unsqueeze(-1), rate.unsqueeze(-1))
        observation = pyro.sample("obs", outcome_dist, obs=y)


def model_gamma(X, y):
    num_features = X.shape[1]
    w = pyro.sample(
        "w", dist.Normal(torch.zeros(num_features), torch.ones(num_features))
    )
    b = pyro.sample("b", dist.Normal(0.0, 1.0))

    y_hat = X.matmul(w) + b
    alpha = pyro.sample("alpha", dist.Gamma(1.0, 1.0))
    beta = pyro.sample("beta", dist.Gamma(1.0, 1.0))

    with pyro.plate("data", len(X)):
        pyro.sample("obs", dist.Gamma(alpha, (beta + 1e-8) / torch.exp(y_hat)), obs=y)
