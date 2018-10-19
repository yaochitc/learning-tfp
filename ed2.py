import tensorflow as tf
import tensorflow_probability.python.edward2 as ed
import tensorflow_probability as tfp

def logistic_regression(features):
    coeffs = ed.Normal(loc=tf.zeros(features.shape[1]), scale=1., name="coeffs")
    intercept = ed.Normal(loc=0., scale=1., name='intercept')
    outcomes = ed.Bernoulli(logits=tf.tensordot(features, coeffs, [[1], [0]]) + intercept,
                            name='outcomes')
    return outcomes


num_features = 55
features = tf.random_normal([100, num_features])
outcomes = tf.random_uniform([100], minval=0, maxval=2, dtype=tf.int32)

log_joint = ed.make_log_joint_fn(logistic_regression)


def target_log_prob_fn(coeffs, intercept):
    return log_joint(features,
                     coeffs=coeffs,
                     intercept=intercept,
                     outcomes=outcomes)

hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target_log_prob_fn,
    step_size=0.1,
    num_leapfrog_steps=5
)
states, kernel_results = tfp.mcmc.sample_chain(
    num_results=1000,
    current_state=[tf.random_normal([55]), tf.random_normal([])],
    kernel=hmc_kernel,
    num_burnin_steps=500
)

with tf.Session() as sess:
    states_, results_ = sess.run([states, kernel_results])