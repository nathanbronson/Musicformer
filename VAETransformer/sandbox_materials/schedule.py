import matplotlib.pyplot as plt
import tensorflow as tf

def make_sig(low, high, center, k, reverse=True):
  return lambda x: tf.divide(high - low, 1 + tf.exp((1 if reverse else -1) * k * (x - center))) + low

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.arg1_eq = make_sig(8 * .0008, 8 * .00314, 35000, 9e-5)#tf.math.rsqrt
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = self.arg1_eq(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def kl_beta(step, period=6000.):#, center=30000., tenthmax=5000.):
  return tf.math.minimum(1, step // period) * ((step % (period - 1.))/period)

learning_rate = CustomSchedule(64)

#.00314-.0008

plt.plot(kl_beta(tf.range(68400, dtype=tf.float32)))
#plt.plot(make_sig(.0008, .00314, 25000, 1e-4)(tf.range(59000, dtype=tf.float32)))
plt.ylabel('Learning Rate')
plt.xlabel('Train Step')

plt.show()

#(8 * .0008, 8 * .00314, 35000, 9e-5)#(8 * .0008, 8 * .00314, 25000, 1e-4)