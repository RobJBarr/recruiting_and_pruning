from conditional_variance import ConditionalVariance
import gpflow
import gpflux
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
def get_model(x_train, Z1, Z2):
    # Layer 1
    kernel1 = gpflow.kernels.SquaredExponential(variance=13.61236537710199, lengthscales=[1.15313208e+02, 1.01911955e+02, 9.58262932e+01, 1.73483535e+02,
       6.07770264e+01, 9.20590890e+01, 9.20590890e+01, 4.39716884e+01,
       4.62773376e+00, 4.22971448e+01, 1.91942880e+01, 2.01140050e+00,
       3.08448762e+01, 9.19957726e-02, 1.04177282e+02, 4.75305642e+01,
       4.64572868e+01])
    inducing_variable1 = gpflow.inducing_variables.InducingPoints(Z1.copy())
    gpflow.set_trainable(inducing_variable1, False)
    gp_layer1 = gpflux.layers.GPLayer(
    kernel1, inducing_variable1, num_data=x_train.shape[0], num_latent_gps=x_train.shape[1], whiten=False
    )
    # Layer 2
    kernel2 = gpflow.kernels.SquaredExponential(variance=0.011844892031544077, lengthscales=[34.39705812, 35.03698464, 93.3435295 , 93.63676621, 44.61998264,
       48.00335988, 48.03797369, 37.13540673, 34.37210917, 31.84380887,
       34.53167441, 34.27607507, 34.30653042, 34.28656465, 66.25545866,
       34.32621139, 34.38317923])
    inducing_variable2 = gpflow.inducing_variables.InducingPoints(Z2.copy())
    gpflow.set_trainable(inducing_variable2, False)
    gp_layer2 = gpflux.layers.GPLayer(
    kernel2,
    inducing_variable2,
    num_data=x_train.shape[0],
    num_latent_gps=x_train.shape[1],
    mean_function=gpflow.mean_functions.Zero(), whiten=False
    )

    # Initialise likelihood and build model
    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.1))
    m = gpflux.models.DeepGP([gp_layer1, gp_layer2], likelihood_layer)
    model = m.as_training_model()
    model.compile(tf.optimizers.Adam(0.01))
    return m, model

from uci_datasets import Dataset
data = Dataset("naval")
x_train, y_train, x_test, y_test = data.get_split(split=5)

kernel1 = gpflow.kernels.SquaredExponential(variance=13.61236537710199, lengthscales=[1.15313208e+02, 1.01911955e+02, 9.58262932e+01, 1.73483535e+02,
       6.07770264e+01, 9.20590890e+01, 9.20590890e+01, 4.39716884e+01,
       4.62773376e+00, 4.22971448e+01, 1.91942880e+01, 2.01140050e+00,
       3.08448762e+01, 9.19957726e-02, 1.04177282e+02, 4.75305642e+01,
       4.64572868e+01])
Z1, _ = ConditionalVariance().compute_initialisation(x_train, 350, kernel1)

kernel2 = gpflow.kernels.SquaredExponential(variance=0.011844892031544077, lengthscales=[34.39705812, 35.03698464, 93.3435295 , 93.63676621, 44.61998264,
       48.00335988, 48.03797369, 37.13540673, 34.37210917, 31.84380887,
       34.53167441, 34.27607507, 34.30653042, 34.28656465, 66.25545866,
       34.32621139, 34.38317923])
Z2, _ = ConditionalVariance().compute_initialisation(x_train, 350, kernel2)

m, _ = get_model(x_train, Z1, Z2)

import tqdm
def train_deep_gp(deep_gp, data, maxiter=int(6000), plotter=None, plotter_interval=10):
    optimizer = tf.optimizers.Adam(0.01)

    @tf.function(autograph=False)
    def objective_closure():
        return -deep_gp.elbo(data)

    @tf.function
    def step():
        optimizer.minimize(objective_closure, deep_gp.trainable_variables)

    tq = tqdm.tqdm(range(maxiter))
    for i in tq:
        step()
        if i % plotter_interval == 0:
            tq.set_postfix_str(f"objective: {objective_closure()}")
    return deep_gp
m = train_deep_gp(m, (x_train, y_train), maxiter=6000)
m_params = gpflow.utilities.traversal.parameter_dict(m)
final_elbo = m.elbo((x_train, y_train))
percentages = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
elbos_before_retrain1 = []
elbos_after_retrain1 = []
elbos_before_retrain2 = []
elbos_after_retrain2 = []
for p in percentages:
    print(p)
    new_model1, _ = get_model(x_train, Z1, Z2)
    gpflow.utilities.multiple_assign(new_model1, m_params)
    Z1_new, _, new_model1, elbos1 = ConditionalVariance().remove_points_deep_gp((x_train, y_train), 0, new_model1, p)
    elbos_before_retrain1.append(elbos1[-1])
    deep_gp1 = train_deep_gp(new_model1, (x_train, y_train), maxiter = 50)
    elbos_after_retrain1.append(new_model1.elbo((x_train, y_train)))

    new_model2, _ = get_model(x_train, Z1, Z2)
    gpflow.utilities.multiple_assign(new_model2, m_params)
    Z2_new, _, new_model2, elbos2 = ConditionalVariance().remove_points_deep_gp((x_train, y_train), 1, new_model1, p)
    elbos_before_retrain2.append(elbos2[-1])
    deep_gp2 = train_deep_gp(new_model2, (x_train, y_train), maxiter = 50)
    elbos_after_retrain2.append(new_model2.elbo((x_train, y_train)))

fig, axs = plt.subplots(1, 2, figsize=(20, 5), dpi=200)
plt.grid()
axs[0].grid(True)
axs[1].grid(True)
fig.suptitle("ELBO of DGP Before and After Retraining: Naval Dataset, 350 Inducing Points in Each Layer")
axs[0].set_title('ELBO of DGP Before and After Retraining First Layer')
axs[0].set_ylabel("ELBO")
axs[0].set_xlabel("Fraction of Original ELBO Before Retraining")
axs[1].set_title('ELBO of DGP Before and After Retraining Second Layer')
axs[1].set_ylabel("ELBO")
axs[1].set_xlabel("Fraction of Original ELBO Before Retraining")

axs[0].plot( [1] + percentages, [final_elbo] + elbos_before_retrain1, label='ELBO Before Retraining')
axs[0].plot( [1] + percentages, [final_elbo] + elbos_after_retrain1, label='ELBO After Retraining')
axs[0].legend(loc='upper left')
axs[1].plot( [1] + percentages, [final_elbo] + elbos_before_retrain2, label='ELBO Before Retraining')
axs[1].plot( [1] + percentages, [final_elbo] + elbos_after_retrain2, label='ELBO After Retraining')
axs[1].legend(loc='upper left')
axs[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
axs[1].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
plt.tight_layout()
fig.savefig(f"/vol/bitbucket/rjb19/recruiting_and_pruning/figs/dgp_remove_and_retrain_naval3.png")

