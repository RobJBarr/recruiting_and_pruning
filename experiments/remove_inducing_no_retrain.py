
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

from uci_datasets import Dataset
data = Dataset("elevators")
x_train, y_train, x_test, y_test = data.get_split(split=5)
sys.stdout.flush()

log_dir_scipy = "logs/scipy/conditional"
lml_task = ScalarToTensorBoard(log_dir_scipy, lambda: full.training_loss(), "training_objective")
monitor = Monitor(
    MonitorTaskGroup([lml_task], period=1)
)
full = gpflow.models.GPR((x_test, y_test), gpflow.kernels.SquaredExponential())
opt = gpflow.optimizers.Scipy()
_ = opt.minimize(full.training_loss, variables=full.trainable_variables, step_callback=monitor)

inducing = x_train[:50,:].copy()
gpflow.kernels.RBF(x_train.shape[0], lengthscales=float(x_train.shape[0])**0.5)
sparse = gpflow.models.SVGP(gpflow.kernels.SquaredExponential(), gpflow.likelihoods.Gaussian(), inducing,whiten=False)


elbo = tf.function(sparse.elbo)
tensor_data = tuple(map(tf.convert_to_tensor, (x_train,y_train)))
elbo(tensor_data)

log_dir_scipy = "logs/scipy/conditional"
sys.stdout.flush()

from scipy.cluster.vq import kmeans
Ms = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000, 1500]
for M in Ms:
    kern = gpflow.kernels.RBF(0.21437344083872883, lengthscales=float(1058.5050695720367))
    Z, _ = ConditionalVariance().compute_initialisation(x_train, M, kern)
    m = gpflow.models.SVGP(
    kernel=kern,
    inducing_variable=Z, likelihood=gpflow.likelihoods.Gaussian())
    lml_task2 = ScalarToTensorBoard("logs/scipy/conditional", lambda: m.elbo((x_train, y_train)), f"elbo-{M}")
    monitor2 = Monitor(
        MonitorTaskGroup([lml_task2], period=1)
    )
    gpflow.set_trainable(m.inducing_variable, False)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss_closure((x_train, y_train)), m.trainable_variables, step_callback=monitor2, options={"maxiter":500})
    models.append(m)
    sys.stdout.flush()
lbs = [m.elbo((x_train, y_train)) for m in models]
cond_var = ConditionalVariance()
params = gpflow.utilities.parameter_dict(models[-1])
new_models = []
final_Z = Z.copy()
final_kern = gpflow.utilities.parameter_dict(kern)
new_Z, indices, gp, new_lbs = ConditionalVariance().remove_points_gp((x_train, y_train), models[-1], 0.9)

lengthscales = [m.kernel.lengthscales.numpy() for m in models]
noise_stds = [m.likelihood.variance.numpy() ** 0.5 for m in models]

fig, axs = plt.subplots(1, 2, figsize=(14, 3), dpi=200)

axs[0].plot(Ms, lbs, label='lower bound')
axs[0].set_title("ELBO as Inducing Points are Added")
axs[0].set_ylabel("Marglik bound (nats)")
axs[0].set_xlabel("Number of inducing points $M$")

axs[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

axs[0].legend(loc='upper right')
plt.tight_layout()

axs[1].legend(loc='upper right')
plt.tight_layout()
axs[1].plot(np.arange(Ms[-1], Ms[-1]-len(new_lbs), step=-1), new_lbs, label='lower bound')
axs[1].invert_xaxis()
axs[1].set_title("ELBO as Inducing Points are Removed")
axs[1].set_ylabel("Marglik bound (nats)")
axs[1].set_xlabel("Number of inducing points $M$")
axs[1].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
fig.savefig(f"/vol/bitbucket/rjb19/recruiting_and_pruning/figs/conditional_remove_inducing_no_training.png")
