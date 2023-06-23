import sys
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import gpflow
from gpflow.utilities import print_summary


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

fig, ax = plt.subplots(3, 6, sharex=False, sharey=False)
fig.set_figheight(12)
fig.set_figwidth(12)
k = 0
for i in range(3):
    for j in range(6):
        col_name = k
        ax[i,j].scatter(x_train[:,k], y_train, s=20, marker='x')
        k += 1
fig.savefig("/vol/bitbucket/rjb19/elevators_plots.png")

log_dir_scipy = "logs/scipy"
lml_task = ScalarToTensorBoard(log_dir_scipy, lambda: full.training_loss(), "training_objective")
monitor = Monitor(
    MonitorTaskGroup([lml_task], period=1)
)


full = gpflow.models.GPR((x_test, y_test), gpflow.kernels.SquaredExponential())
opt = gpflow.optimizers.Scipy()
_ = opt.minimize(full.training_loss, variables=full.trainable_variables, step_callback=monitor, options = {"maxiter": 80})


inducing = x_train[:50,:].copy()
kern = gpflow.kernels.SquaredExponential
sparse = gpflow.models.SVGP(gpflow.kernels.SquaredExponential(), gpflow.likelihoods.Gaussian(), inducing,whiten=False)


elbo = tf.function(sparse.elbo)
tensor_data = tuple(map(tf.convert_to_tensor, (x_train,y_train)))
elbo(tensor_data)


minibatch_size = 10000

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat().shuffle(500000)

train_iter = iter(train_dataset.batch(minibatch_size))

ground_truth = elbo(tensor_data).numpy()




ground_truth



elbo(next(train_iter))


import itertools
evals = [elbo(minibatch).numpy() for minibatch in itertools.islice(train_iter, 1000)]


fig = plt.figure()
plt.hist(evals, label="Minibatch estimations")
plt.axvline(ground_truth, c="k", label="Ground truth")
plt.axvline(np.mean(evals), c="g", ls="--", label="Minibatch mean")
plt.legend()
plt.title("Histogram of ELBO evaluations using minibatches")
fig.savefig("/vol/bitbucket/rjb19/elbo_histograms.png")
()


import time
N = 10000
minibatch_proportions = np.logspace(-2, 0, 10)
times = []
objs = []
for mbp in minibatch_proportions:
    batchsize = int(N * mbp)
    train_iter = iter(train_dataset.batch(batchsize))
    start_time = time.time()
    objs.append([elbo(minibatch) for minibatch in itertools.islice(train_iter, 20)])
    times.append(time.time() - start_time)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.plot(minibatch_proportions, times, "x-")
ax1.set_xlabel("Minibatch proportion")
ax1.set_ylabel("Time taken")
ax2.axhline(ground_truth, c="k", label="Ground truth")
ax2.plot(minibatch_proportions, np.array(objs), "kx")
ax2.set_xlabel("Minibatch proportion")
ax2.set_ylabel("ELBO estimates")
f.savefig("/vol/bitbucket/rjb19/minibatch_elbos.png")

minibatch_size = 10000

def run_adam(model, iterations):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam()

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        if step % 10 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
    return logf


maxiter = 10000

logf = run_adam(sparse, maxiter)
fig = plt.figure()
plt.plot(np.arange(maxiter)[::10], logf)
plt.xlabel("iteration")
_ = plt.ylabel("ELBO")
fig.savefig("/vol/bitbucket/rjb19/iters_vs_elbo.png")

log_dir_scipy = "logs/scipy"



from scipy.cluster.vq import kmeans
Ms = [1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 50, 100, 200, 300, 500, 750, 1000, 1250, 1500,2000, 3000]
models = []
for M in Ms:
    Z, _ = kmeans(x_train,M)
    m = gpflow.models.SGPR(
    (x_train, y_train),
    kernel=gpflow.kernels.RBF(x_train.shape[0], lengthscales=float(x_train.shape[0])**0.5),
    inducing_variable=Z)
    lml_task2 = ScalarToTensorBoard("logs/scipy", lambda: m.elbo(), f"elbo-{M}")
    monitor2 = Monitor(
        MonitorTaskGroup([lml_task2], period=1)
    )
    gpflow.set_trainable(m.inducing_variable, True)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables, step_callback=monitor2, options={"maxiter":500})
    models.append(m)
    print(M)
    sys.stdout.flush()


x_train.shape


lbs = [m.elbo() for m in models]
ubs = [m.upper_bound() for m in models]
lengthscales = [m.kernel.lengthscales.numpy() for m in models]
noise_stds = [m.likelihood.variance.numpy() ** 0.5 for m in models]


lbs


lml = full.log_marginal_likelihood()
fig, axs = plt.subplots(1, 2, figsize=(14, 3), dpi=200)

axs[0].plot(Ms, lbs, label='lower bound')
axs[0].plot(Ms, ubs, label='upper bound')
axs[0].set_title("Tightness of bounds")
axs[0].set_ylabel("Marglik bound (nats)")
axs[0].set_xlabel("Number of inducing points $M$")
axs[0].legend(loc='upper right')
axs[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

if len(axs) > 2:
    axs[1].plot(Ms, lengthscales)
    ax2 = axs[1].twinx()
    ax2.plot(Ms, noise_stds, 'C1')
    ax2.set_ylabel("noise std", color="C1")
    axs[1].set_ylabel("lengthscale", color="C0")
    axs[1].set_xlabel("Number of inducing points $M$")
    axs[1].set_title("Hyperparameters")
    axs[1].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

plt.tight_layout()
fig.savefig(f"/vol/bitbucket/rjb19/elevators_tightness.png")