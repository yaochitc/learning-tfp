import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def make_dataset(n, d, link, scale=1., dtype=np.float32):
    model_coefficients = tfd.Uniform(
        low=np.array(-1, dtype),
        high=np.array(1, dtype)).sample(d, seed=42)
    radius = np.sqrt(2.)
    model_coefficients *= radius / tf.linalg.norm(model_coefficients)
    model_matrix = tfd.Normal(
        loc=np.array(0, dtype),
        scale=np.array(1, dtype)).sample([n, d], seed=43)
    scale = tf.convert_to_tensor(scale, dtype)
    linear_response = tf.tensordot(
        model_matrix, model_coefficients, axes=[[1], [0]])
    if link == 'linear':
        response = tfd.Normal(loc=linear_response, scale=scale).sample(seed=44)
    elif link == 'probit':
        response = tf.cast(
            tfd.Normal(loc=linear_response, scale=scale).sample(seed=44) > 0,
            dtype)
    elif link == 'logit':
        response = tfd.Bernoulli(logits=linear_response).sample(seed=44)
    else:
        raise ValueError('unrecognized true link: {}'.format(link))
    return model_matrix, response, model_coefficients


X, Y, w_true = make_dataset(n=int(1e6), d=100, link='probit')

w, linear_response, is_converged, num_iter = tfp.glm.fit(
    model_matrix=X,
    response=Y,
    model=tfp.glm.BernoulliNormalCDF())
log_likelihood = tfp.glm.BernoulliNormalCDF().log_prob(Y, linear_response)

with tf.Session() as sess:
    [w_, linear_response_, is_converged_, num_iter_, Y_, w_true_,
     log_likelihood_] = sess.run([
        w, linear_response, is_converged, num_iter, Y, w_true,
        log_likelihood])

print('is_converged: ', is_converged_)
print('    num_iter: ', num_iter_)
print('    accuracy: ', np.mean((linear_response_ > 0.) == Y_))
print('    deviance: ', 2. * np.mean(log_likelihood_))
print('||w0-w1||_2 / (1+||w0||_2): ', (np.linalg.norm(w_true_ - w_, ord=2) /
                                       (1. + np.linalg.norm(w_true_, ord=2))))