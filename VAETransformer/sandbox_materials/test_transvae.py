from tf_transvae import VAEEncoder, VAEDecoder
import tensorflow as tf

input_data = tf.random.normal(shape=(4, 1290, 64))

l = VAEEncoder(4, 4, 64, 128, 32, False, 1)
l2 = VAEDecoder(4, 4, 64, 128, 32, False)

r, mu, logvar = l(input_data)
r2 = l2(input_data, r)

print(r.shape, r2.shape)

from tf_transvae import EncoderDecoder
import tensorflow as tf

input_data = tf.random.normal(shape=(4, 1290, 64))

l = EncoderDecoder(64, 4, 4, 4, 128, 32, False, 1)

r = l((input_data, input_data))

print(*[_r.shape for _r in r])