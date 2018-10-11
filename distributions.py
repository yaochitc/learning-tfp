import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

tfe = tf.contrib.eager
try:
    tfe.enable_eager_execution()
except ValueError:
    pass

n = tfd.Normal(loc=0., scale=1.)

nd = tfd.MultivariateNormalDiag(loc=[0., 10.], scale_diag=[1., 4.])

b3 = tfd.Bernoulli(probs=[.3, .5, .7])

b3_joint = tfd.Independent(b3, reinterpreted_batch_ndims=1)

print(b3_joint.prob([1, 1, 0]))