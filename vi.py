import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

p = tfd.Bernoulli(0.4)
q = tfd.Bernoulli(0.6)

print(p)

exact_kl_bernoulli_bernoulli = tfp.distributions.kl_divergence(p, q)

approx_kl_p_q = tfp.vi.monte_carlo_csiszar_f_divergence(
    f = tfp.vi.kl_reverse,
    p_log_prob=q.log_prob,
    q = p,
    num_draws= int(1e5)
)

with tf.Session() as sess:
    [exact_kl_bernoulli_bernoulli_, approx_kl_p_q_] = \
        sess.run([exact_kl_bernoulli_bernoulli, approx_kl_p_q])

print('exact_kl_bernoulli_bernoulli: ', exact_kl_bernoulli_bernoulli_)
print('approx_kl_p_q: ', approx_kl_p_q_)