# %% 
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

# %%
data = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()
x_train = np.pad(x_train, ((0,0), (2,2), (2,2)), 'constant', constant_values=0)
x_train = x_train / 255.
x_test = np.pad(x_test, ((0,0), (2,2), (2,2)), 'constant', constant_values=0)
x_test = x_test / 255.

# %%
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# %%
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=6, kernel_size=(3,3), activation='relu', input_shape=(32, 32, 1)),
    keras.layers.AveragePooling2D(),
    keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'),
    keras.layers.AveragePooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(120, activation='relu'),
    keras.layers.Dense(84, activation='relu'),
    keras.layers.Dense(10)])

# %%
predictions = model(x_train[:1]).numpy()
predictions

# %%
tf.nn.softmax(predictions).numpy()

# %%
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# %%
model.compile(
    optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy'])
# %%
model.fit(x_train, y_train, epochs=10)

# %%
model.evaluate(x_test,  y_test, verbose=2)

# %%
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# %%
probability_model(x_test[:5])

# %%
