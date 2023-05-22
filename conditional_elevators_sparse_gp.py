
import sys
print("Beginning script")
sys.stdout.flush()
import pickle
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import gpflow
from gpflow.utilities import print_summary
from tools import datasets
from tools import plotting
import abc
from typing import Optional, Callable

print("Here ")
sys.stdout.flush()

class ConditionalVariance():
    def __init__(self, sample: Optional[bool] = False, threshold: Optional[int] = 0.0, seed: Optional[int] = 0,
                 **kwargs):
        """
        :param sample: bool, if True, sample points into subset to use with weights based on variance, if False choose
        point with highest variance at each iteration
        :param threshold: float or None, if not None, if tr(Kff-Qff)<threshold, stop choosing inducing points as the approx.
        has converged.
        """
        self.randomized=True
        self.seed = seed if self.randomized else None
        self.sample = sample
        self.threshold = threshold

    def compute_initialisation(self, training_inputs: np.ndarray, M: int,
                               kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray]):
        N = training_inputs.shape[0]
        print(N)
        perm = np.random.permutation(N)  # permute entries so tiebreaking is random
        training_inputs = training_inputs[perm]
        # note this will throw an out of bounds exception if we do not update each entry
        indices = np.zeros(M, dtype=int) + N
        di = kernel(training_inputs, None, full_cov=False) + 1e-12  # jitter
        if self.sample:
            indices[0] = sample_discrete(di)
        else:
            indices[0] = np.argmax(di)  # select first point, add to index 0
        if M == 1:
            indices = indices.astype(int)
            Z = training_inputs[indices]
            indices = perm[indices]
            return Z, indices
        ci = np.zeros((M - 1, N))  # [M,N]
        for m in range(M - 1):
            j = int(indices[m])  # int
            new_Z = training_inputs[j:j + 1]  # [1,D]
            dj = np.sqrt(di[j])  # float
            cj = ci[:m, j]  # [m, 1]
            Lraw = np.array(kernel(training_inputs, new_Z, full_cov=True))
            L = np.round(np.squeeze(Lraw), 20)  # [N]
            L[j] += 1e-12  # jitter
            ei = (L - np.dot(cj, ci[:m])) / dj
            ci[m, :] = ei
            try:
                di -= ei ** 2
            except FloatingPointError:
                pass
            di = np.clip(di, 0, None)
            if self.sample:
                indices[m + 1] = sample_discrete(di)
            else:
                indices[m + 1] = np.argmax(di)  # select first point, add to index 0
            # sum of di is tr(Kff-Qff), if this is small things are ok
            print(di.shape, m)
            if np.sum(np.clip(di, 0, None)) < self.threshold:
                indices = indices[:m]
                warnings.warn("ConditionalVariance: Terminating selection of inducing points early.")
                break
        indices = indices.astype(int)
        Z = training_inputs[indices]
        indices = perm[indices]
        return Z, indices

    def __repr__(self):
        params = ', '.join([f'{k}={v}' for k, v in self.__dict__.items()
                            if
                            k not in ['_randomized'] and
                            not (k == "threshold" and self.threshold == 0.0)])
        return f"{type(self).__name__}({params})"

# %%
from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)

# %%
from uci_datasets.uci_datasets import Dataset
data = Dataset("elevators")
x_train, y_train, x_test, y_test = data.get_split(split=5)
sys.stdout.flush()

log_dir_scipy = "logs/scipy"
lml_task = ScalarToTensorBoard(log_dir_scipy, lambda: full.training_loss(), "training_objective")
monitor = Monitor(
    MonitorTaskGroup([lml_task], period=1)
)
full = gpflow.models.GPR((x_test, y_test), gpflow.kernels.SquaredExponential())
opt = gpflow.optimizers.Scipy()
_ = opt.minimize(full.training_loss, variables=full.trainable_variables, step_callback=monitor, options = {"maxiter": 80})

log_dir = "/vol/bitbucket/rjb19"
ckpt = tf.train.Checkpoint(full=full)
manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=5)
print("here 43" , log_dir)
print("asdfasdfasdfasdfasdfasdf")
sys.stdout.flush()
print(manager.save())
sys.stdout.flush()

inducing = x_train[:50,:].copy()
kern = gpflow.kernels.SquaredExponential
sparse = gpflow.models.SVGP(gpflow.kernels.SquaredExponential(), gpflow.likelihoods.Gaussian(), inducing,whiten=False)


elbo = tf.function(sparse.elbo)
tensor_data = tuple(map(tf.convert_to_tensor, (x_train,y_train)))
elbo(tensor_data)

# %%
log_dir_scipy = "logs/scipy"
sys.stdout.flush()
# %%
from scipy.cluster.vq import kmeans
Ms = [3000]
models = []
for M in Ms:
    
    kern = gpflow.kernels.RBF(x_train.shape[1], lengthscales=float(x_train.shape[1])**0.5)
    Z, _ = ConditionalVariance().compute_initialisation(x_train, M, kern)
    m = gpflow.models.SGPR(
    (x_train, y_train),
    kernel=kern,
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
params = gpflow.utilities.parameter_dict(m.kernel)
import pickle
with open("kernel.pickle", "wb") as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%
lbs = [m.elbo() for m in models]
ubs = [m.upper_bound() for m in models]
lengthscales = [m.kernel.lengthscales.numpy() for m in models]
noise_stds = [m.likelihood.variance.numpy() ** 0.5 for m in models]

# %%
lml = full.log_marginal_likelihood()

# %%

fig, axs = plt.subplots(1, 1, figsize=(14, 3), dpi=200)

axs.plot(Ms, lbs, label='lower bound')
axs.plot(Ms, ubs, label='upper bound')
#axs[0].axvline(Ms[i], color='C3')
axs.set_title("Tightness of bounds")
axs.set_ylabel("Marglik bound (nats)")
axs.set_xlabel("Number of inducing points $M$")
# axs[0].axhline(lml, color='k', linestyle='--', label="marg. lik.", linewidth=2.0)
axs.legend(loc='upper right')
axs.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

plt.tight_layout()
fig.savefig(f"/vol/bitbucket/rjb19/conditional_elevators_tightness.png")

m = models[-1]
log_dir = "/vol/bitbucket/rjb19"
ckpt = tf.train.Checkpoint(m=m)
manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=5)
print(manager.save())
sys.stdout.flush()
# saver.save('/vol/bitbucket/rjb19/elevators_conditional_model', m)
