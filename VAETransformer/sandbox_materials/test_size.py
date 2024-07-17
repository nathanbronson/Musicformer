import tensorflow as tf

"""
64 -> 32: 32 use 8 for 4
1290 -> 32: 1258
"""

input_shape = (4, 1290, 64)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv1D(128, 3)(x)
y = tf.keras.layers.BatchNormalization()(y)
y = tf.keras.activations.relu(y)
y = tf.keras.layers.Conv1D(256, 5, 2)(y)
y = tf.keras.layers.BatchNormalization()(y)
y = tf.keras.activations.relu(y)
y = tf.keras.layers.Conv1D(512, 8, 4)(y)
y = tf.keras.layers.BatchNormalization()(y)
y = tf.keras.activations.relu(y)
y = tf.keras.layers.Conv1D(512, 8, 4)(y)
y = tf.keras.layers.BatchNormalization()(y)
y = tf.keras.activations.relu(y)
y = tf.keras.layers.Flatten()(y) #WHERE TO GET SHAPE
y = tf.keras.layers.Dense(512)(y)
y = tf.keras.layers.BatchNormalization()(y)
y = tf.keras.activations.relu(y)
y = tf.keras.layers.Dense(64)(y)
#y = (y)

print(y.shape)

latent_shape = (4, 64)
x = tf.random.normal(latent_shape)
y = tf.keras.layers.Dense(512)(x)
y = tf.keras.layers.BatchNormalization()(y)
y = tf.keras.activations.relu(y)
y = tf.keras.layers.Dense(19456)(y)  #NEED FLATTEN SHAPE
y = tf.keras.layers.Reshape((38, 512))(y)
y = tf.keras.layers.Conv1DTranspose(512, 8, 4)(y)
y = tf.keras.layers.BatchNormalization()(y)
y = tf.keras.activations.relu(y)
y = tf.keras.layers.Conv1DTranspose(512, 8, 4)(y)
y = tf.keras.layers.BatchNormalization()(y)
y = tf.keras.activations.relu(y)
y = tf.keras.layers.Conv1DTranspose(256, 5, 2)(y)
y = tf.keras.layers.BatchNormalization()(y)
y = tf.keras.activations.relu(y)
y = tf.keras.layers.Conv1DTranspose(128, 3)(y)
y = tf.keras.layers.BatchNormalization()(y)
y = tf.keras.activations.relu(y)
y = tf.keras.layers.Conv1DTranspose(64, 31)(y)
y = tf.keras.layers.BatchNormalization()(y)
y = tf.keras.activations.relu(y)
print(y.shape)

exit()
























"""
y = tf.keras.layers.Conv2D(2, (3, 3), activation="relu", padding="same")(x)
y = tf.keras.layers.MaxPooling2D((2, 1))(y)
y = tf.keras.layers.Conv2D(4, (3, 3), activation="relu", padding="same")(y)
y = tf.keras.layers.MaxPooling2D((2, 1))(y)
y = tf.keras.layers.Conv2D(8, (5, 3), activation="relu", padding="same")(y)
y = tf.keras.layers.MaxPooling2D((2, 1))(y)
y = tf.keras.layers.Conv2D(16, (5, 3), activation="relu")(y)
y = tf.keras.layers.MaxPooling2D((2, 2))(y)
y = tf.keras.layers.Conv2D(32, (5, 5))(y)
y = tf.keras.layers.MaxPooling2D((3, 3))(y)
y = tf.keras.layers.Flatten()(y)
#y = tf.keras.layers.Conv2D(2, 3, strides=(2, 1), input_shape=input_shape[1:], activation="relu", padding="same")(y)
#y = tf.keras.layers.Conv2D(2, 3, strides=(2, 1), input_shape=input_shape[1:], activation="relu", padding="same")(y)
#y = tf.keras.layers.Conv2D(2, (314, 8), input_shape=input_shape[1:], activation="relu")(x)
#y = tf.keras.layers.Conv2D(4, (314, 8), input_shape=input_shape[1:], activation="relu")(y)
#y = tf.keras.layers.Conv2D(8, (314, 8), input_shape=input_shape[1:], activation="relu")(y)
#y = tf.keras.layers.Conv2D(16, (314, 8), input_shape=input_shape[1:], activation="relu")(y)
#y = tf.keras.layers.Conv2D(32, 3, input_shape=input_shape[1:], activation="relu")(y)

print(y.shape)

x = tf.random.normal((4, 64))
y = tf.keras.layers.Dense(6912)(x)
y = 

\"\"\"
down_layer_list = [
    tf.keras.layers.Conv2D(2, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 1)),
    tf.keras.layers.Conv2D(4, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 1)),
    tf.keras.layers.Conv2D(8, (5, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 1)),
    tf.keras.layers.Conv2D(16, (5, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (5, 5)),
    tf.keras.layers.MaxPooling2D((3, 3)),
    tf.keras.layers.Flatten()
]
\"\"\"
"""