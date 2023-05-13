import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from torch.utils.data import TensorDataset, DataLoader
import pyro.nn as pnn
from sklearn.preprocessing import StandardScaler


class RegressionModel(pnn.PyroModule):
    def __init__(self, input_features, output_features=1):
        super().__init__()
        self.linear = pnn.PyroModule[torch.nn.Linear](input_features, output_features)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x)).squeeze(-1)


class BayesianRegressionModel(pnn.PyroModule):
    def __init__(self, in_features, out_features=1):
        super().__init__()
        self.linear = pnn.PyroModule[torch.nn.Linear](in_features, out_features)
        self.linear.weight = pnn.PyroSample(
            dist.Normal(0.0, 1.0).expand([out_features, in_features]).to_event(2)
        )
        self.linear.bias = pnn.PyroSample(
            dist.Normal(0.0, 10.0).expand([out_features]).to_event(0)
        )

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0.0, 10.0))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean


class BayesianRegressionModel2(pnn.PyroModule):
    def __init__(self, num_features):
        super().__init__()
        self.linear = pnn.PyroModule[torch.nn.Linear](num_features, 1)
        self.sigmoid = torch.nn.Sigmoid()

        # Define priors
        self.linear.weight = pnn.PyroSample(
            dist.Normal(
                torch.zeros(num_features, 1), torch.ones(num_features, 1)
            ).to_event(2)
        )
        self.linear.bias = pnn.PyroSample(
            dist.Normal(torch.tensor([0.0]), torch.tensor([10.0]))
        )

    def model(self, x_data, y_data):
        with pyro.plate("data", len(x_data)):
            prediction_mean = self.forward(x_data).squeeze()
            pyro.sample(
                "obs",
                dist.Normal(prediction_mean, 1.0),
                obs=y_data.squeeze(1),
            )

    def guide(self, x_data, y_data):
        weight_loc = pyro.param("weight_loc", torch.randn(self.linear.in_features, 1))
        weight_scale = pyro.param(
            "weight_scale",
            torch.ones(self.linear.in_features, 1),
            constraint=dist.constraints.positive,
        )
        bias_loc = pyro.param("bias_loc", torch.randn(1))
        bias_scale = pyro.param(
            "bias_scale", torch.tensor(1.0), constraint=dist.constraints.positive
        )

        weight_dist = dist.Normal(weight_loc, weight_scale).to_event(2)
        bias_dist = dist.Normal(bias_loc, bias_scale)

        self.linear.weight = pnn.PyroSample(weight_dist)
        self.linear.bias = pnn.PyroSample(bias_dist)

        # with pyro.plate("data", len(x_data)):
        prediction_mean = self.forward(x_data).squeeze()

    def forward(self, x):
        return self.linear(x).squeeze(-1)


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
