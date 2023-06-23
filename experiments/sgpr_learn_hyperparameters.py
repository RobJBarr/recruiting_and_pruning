import sys
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import gpflow
from gpflow.utilities import print_summary
from conditional_variance import ConditionalVariance
from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)
dataset_name = sys.argv[1]
from uci_datasets import Dataset
data = Dataset(dataset_name)
x_train, y_train, x_test, y_test = data.get_split(split=5)

kern = gpflow.kernels.SquaredExponential()
Z, _ = ConditionalVariance().compute_initialisation(x_train, 5000, kern)
m = gpflow.models.SGPR((x_train, y_train),
kernel=kern,
inducing_variable=Z, likelihood=gpflow.likelihoods.Gaussian())
lml_task = ScalarToTensorBoard("logs/scipy", lambda: m.elbo(), f"elbo-{5000}-{dataset_name}")
monitor = Monitor(
    MonitorTaskGroup([lml_task], period=1)
)
gpflow.set_trainable(m.inducing_variable, False)

opt = gpflow.optimizers.Scipy()
opt.minimize(m.training_loss, m.trainable_variables, step_callback=monitor)


import json
lengthscales = kern.lengthscales.numpy().tolist()
variance = kern.variance.numpy().tolist()
kernel_hyperparams = {"lengthscales": lengthscales, "variance": variance}
with open(f"hyperparameters/sgpr_{dataset_name}_hyperparams.json", "w") as f:
    json.dump(kernel_hyperparams, f)