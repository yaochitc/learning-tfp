import tensorflow as tf
import tensorflow_probability.python.edward2 as ed
import tensorflow_probability as tfp
import numpy as np


def probabilistic_pca(data_dim, latent_dim, num_datapoints, stddv_datapoints):  # (unmodeled) data
    w = ed.Normal(loc=tf.zeros([data_dim, latent_dim]),
                  scale=2.0 * tf.ones([data_dim, latent_dim]),
                  name="w")  # parameter
    z = ed.Normal(loc=tf.zeros([latent_dim, num_datapoints]),
                  scale=tf.ones([latent_dim, num_datapoints]),
                  name="z")  # parameter
    x = ed.Normal(loc=tf.matmul(w, z),
                  scale=stddv_datapoints * tf.ones([data_dim, num_datapoints]),
                  name="x")  # (modeled) data
    return x, (w, z)


log_joint = ed.make_log_joint_fn(probabilistic_pca)

num_datapoints = 5000
data_dim = 2
latent_dim = 1
stddv_datapoints = 0.5

model = probabilistic_pca(data_dim=data_dim,
                          latent_dim=latent_dim,
                          num_datapoints=num_datapoints,
                          stddv_datapoints=stddv_datapoints)

with tf.Session() as sess:
  x_train, (actual_w, actual_z) = sess.run(model)

tf.reset_default_graph()

w = tf.Variable(np.ones([data_dim, latent_dim]), dtype=tf.float32)
z = tf.Variable(np.ones([latent_dim, num_datapoints]), dtype=tf.float32)

def target(w, z):
  """Unnormalized target density as a function of the parameters."""
  return log_joint(data_dim=data_dim,
                   latent_dim=latent_dim,
                   num_datapoints=num_datapoints,
                   stddv_datapoints=stddv_datapoints,
                   w=w, z=z, x=x_train)

energy = -target(w, z)

optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
train = optimizer.minimize(energy)

init = tf.global_variables_initializer()

t = []

num_epochs = 200

with tf.Session() as sess:
  sess.run(init)

  for i in range(num_epochs):
    sess.run(train)
    if i % 5 == 0:
      cE, cw, cz = sess.run([energy, w, z])
      t.append(cE)

  w_inferred_map = sess.run(w)
  z_inferred_map = sess.run(z)