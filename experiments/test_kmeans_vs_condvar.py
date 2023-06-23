
import sys
from conditional_variance import ConditionalVariance
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import gpflow
from gpflow.utilities import print_summary
from scipy.cluster.vq import kmeans

from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)

from uci_datasets import Dataset
data = Dataset("elevators")
x_train, y_train, x_test, y_test = data.get_split(split=5)

Ms = [1,2, 3, 4, 5, 6, 7, 8, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000]
models = []
for M in Ms:
    kern = gpflow.kernels.RBF(variance=16467.93166587576, lengthscales=740.4639311514006)
    Z, _ = ConditionalVariance().compute_initialisation(x_train, M, kern)
    m = gpflow.models.SGPR(
    (x_train, y_train),
    kernel=kern,
    inducing_variable=Z)
    lml_task2 = ScalarToTensorBoard("logs/scipy/conditional", lambda: m.elbo(), f"elbo-{M}")
    monitor2 = Monitor(
        MonitorTaskGroup([lml_task2], period=1)
    )
    gpflow.set_trainable(m.inducing_variable, False)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables, step_callback=monitor2, options={"maxiter":500})
    models.append(m)
    print(M)
    sys.stdout.flush()
cond_var = ConditionalVariance()
params = gpflow.utilities.parameter_dict(models[-1])
new_models = []
final_Z = Z.copy()
final_kern = gpflow.utilities.parameter_dict(kern)
i = 0
for M in Ms:
    kern = gpflow.kernels.RBF(16453.294046883326, lengthscales=float(1058.5050695720367))
    Z = kmeans(x_train, M)[0]
    sys.stdout.flush()
    m = gpflow.models.SGPR(
    (x_train, y_train),
    kernel=kern,
    inducing_variable=Z)
    lml_task2 = ScalarToTensorBoard("logs/scipy/conditional", lambda: m.elbo(), f"elbo-{M}-kmeans")
    monitor2 = Monitor(
        MonitorTaskGroup([lml_task2], period=1)
    )
    gpflow.set_trainable(m.inducing_variable, False)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables, step_callback=monitor2, options={"maxiter":500})
    new_models.append(m)
    print(M)
    sys.stdout.flush()

new_lbs = [m.elbo() for m in new_models]
new_ubs = [m.upper_bound() for m in new_models]
new_lengthscales = [m.kernel.lengthscales.numpy() for m in new_models]
new_noise_stds = [m.likelihood.variance.numpy() ** 0.5 for m in new_models]

lbs = [m.elbo() for m in models]
ubs = [m.upper_bound() for m in models]
lengthscales = [m.kernel.lengthscales.numpy() for m in models]
noise_stds = [m.likelihood.variance.numpy() ** 0.5 for m in models]

fig, axs = plt.subplots(1, 1, figsize=(14, 3), dpi=200)

axs.plot(Ms, lbs, label='lower bound conditional')
axs.plot(Ms, ubs, label='upper bound conditional')
axs.set_title("Tightness of bounds")
axs.set_ylabel("Marglik bound (nats)")
axs.set_xlabel("Number of inducing points $M$")
axs.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
axs.plot(Ms, new_lbs, label='lower bound removing kmeans')
axs.plot(Ms, new_ubs, label='upper bound removing kmeans')
axs.legend(loc='upper right')
plt.tight_layout()
fig.savefig(f"/vol/bitbucket/rjb19/conditional_vs_kmeans.png")
