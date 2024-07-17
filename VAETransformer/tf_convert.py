from glob import glob
from tqdm import tqdm
from pickle import load
import tensorflow as tf
from functools import partial
import os
import tensorflow_datasets as tfds

BUFFER_SIZE = 5000
d_model = 64
BATCH_SIZE = 7
DO_CONVERT = True

files = glob("./spectrograms/mel_*.pkl")
print(len(files))

generator_output_signature = tf.TensorSpec(shape=(1290, d_model), dtype=tf.float32)

def load_file(file):
  with open(file, "rb") as doc:
    d = load(doc)
  return d

def prepare_batch(dat):###########not actually batch
    context = dat#.to_tensor()

    inputs = dat[:-1]#.to_tensor()  # Drop the [END] tokens
    labels = dat[1:]#.to_tensor()   # Drop the [START] tokens

    return (context, inputs), labels

def gen(files, use_tqdm=False):
  for file in (tqdm(files) if use_tqdm else files):
    yield tf.convert_to_tensor(load_file(file), dtype=tf.float32)

def make_convert_batches(fdat):
  return (
      tf.data.Dataset.from_generator(partial(gen, fdat, use_tqdm=True), output_signature=generator_output_signature)
      .shuffle(BUFFER_SIZE)
      .map(prepare_batch, tf.data.AUTOTUNE)
      .prefetch(buffer_size=tf.data.AUTOTUNE)
  )

tfrecords_dir = "./tfrecords"
num_samples = 300
num_tfrecords = len(files) // num_samples
if len(files) % num_samples:
    num_tfrecords += 1  # add one record if there are any remaining samples

if not os.path.exists(tfrecords_dir):
    os.makedirs(tfrecords_dir)  # creating TFRecords output folder

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _parse_tfr_element(element):
  parse_dic = {
    'pt': tf.io.FixedLenFeature([], tf.string), # Note that it is tf.string, not tf.float32
  }
  example_message = tf.io.parse_single_example(element, parse_dic)

  _pt = tf.io.parse_tensor(example_message['pt'], out_type=tf.float32) # get byte string
  #feature = tf.io.parse_tensor(_pt, out_type=tf.float32) # restore 2D array from byte string
  #feature = tf.io.parse_tensor(b_feature, out_type=tf.float32) # restore 2D array from byte string
  return (_pt, tf.concat([tf.zeros_like(_pt[:1]), _pt], axis=0)), tf.concat([_pt, tf.zeros_like(_pt[:1])], axis=0)

if DO_CONVERT:
    accumulator = []
    tfrec_num = 0
    for (pt, _), _ in make_convert_batches(files):
        accumulator.append(
            tf.train.Example(features=tf.train.Features(feature={
                "pt": _bytes_feature(serialize_array(pt)),#tf.train.Feature(float_list=tf.train.FloatList(value=pt)),
            })).SerializeToString()
        )
        if len(accumulator) >= num_samples:
            with tf.io.TFRecordWriter(tfrecords_dir + "/file_{}.tfrec".format(str(tfrec_num))) as writer:
                for a in accumulator:
                    writer.write(a)
            accumulator = []
            tfrec_num += 1

train_batches = (tf.data.TFRecordDataset(glob(tfrecords_dir + "/*.tfrec"), num_parallel_reads=tf.data.AUTOTUNE)
                .map(_parse_tfr_element, num_parallel_calls=tf.data.AUTOTUNE)
                .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

pt = None
en = None
for (pt, en), enl in train_batches.take(1):
  break

print(pt.shape, en.shape, enl.shape)

tfds.benchmark(train_batches, batch_size=BATCH_SIZE)