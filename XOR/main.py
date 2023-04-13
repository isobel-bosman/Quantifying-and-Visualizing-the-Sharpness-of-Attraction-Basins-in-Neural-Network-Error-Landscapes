import numpy
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import random
import math
import matplotlib.pyplot as plt


def get_weight_count(w):
    # Count all the bits in the array
    c = 0
    for i, _list in enumerate(w):
        for j, sub_list in enumerate(_list):
            try:
                len(sub_list)
                c += len(sub_list)
            except:
                c += 1
    return c


def get_index_subset(c):
    # Find indices of 50% of the count
    indices = list(range(c))
    if (c % 2) == 0:
        half = c / 2
    else:
        addition = random.randint(0, 1)
        half = math.floor(c / 2)
        half += addition

    subset = random.sample(indices, half)
    return subset


def modify_weights(arr, c, amount):
    # Get chosen indices
    indices = get_index_subset(c)
    # print(indices)

    # Modify weights with chosen indices
    index = -1
    for i, weight_list in enumerate(arr):
        for j, sub_list in enumerate(weight_list):
            try:
                len(sub_list)
                for k, sub_sub_list in enumerate(sub_list):
                    index += 1
                    if index in indices:
                        # print("chosen", index)
                        arr[i][j][k] += amount
                    # print('a[{}][{}][{}] = {}'.format(i, j, k, sub_sub_list))
            except:
                index += 1
                if index in indices:
                    # print("chosen", index)
                    arr[i][j] += amount
                # print('a[{}][{}] = {}'.format(i, j, sub_list))
    return arr

# Create the base model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(units=2, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_train = tf.cast([[0, 0], [0, 1], [1, 0], [1, 1]], tf.float32)
y_train = tf.cast([0, 1, 1, 0], tf.float32)

callbacks = [EarlyStopping(monitor='loss', patience=200),
             ModelCheckpoint(filepath='best_model.h5', monitor='loss', save_weights_only=True, save_best_only=True)]

history = model.fit(
    x=x_train, y=y_train,
    epochs=2500,
    steps_per_epoch=1,
    callbacks=callbacks,
    verbose=0)

# Find the best loss value
print('Evaluate weights')
model.load_weights('best_model.h5')
evalModel = model.evaluate(x_train, y_train)
loss = evalModel[0]
normalLoss = [loss]

print("Original weights")
weights = np.array(model.get_weights())
print(weights)
print()
print()

count = get_weight_count(weights)

# Generate Losses for Boxplot
subtractedLoss = []
for i in range(10):
    modWeights = modify_weights(weights, count, -0.01)
    model.set_weights(modWeights)
    evalModel = model.evaluate(x_train, y_train)
    loss = evalModel[0]
    subtractedLoss.append(loss)

addedLoss = []
for i in range(10):
    modWeights = modify_weights(weights, count, 0.01)
    model.set_weights(modWeights)
    evalModel = model.evaluate(x_train, y_train)
    loss = evalModel[0]
    addedLoss.append(loss)

print("subtractedLoss", subtractedLoss)
print("normal", normalLoss)
print("addedLoss", addedLoss)

data = [subtractedLoss, normalLoss, addedLoss]

# Create Boxplot
fig1, ax1 = plt.subplots()
ax1.set_title('Loss Distribution for XOR Dataset')
ax1.boxplot(data)
plt.xticks([1, 2, 3], ['-0.01', '0', '+0.01'])
plt.xlabel('Modification amount for a random sample of the weights')
plt.ylabel('Loss')
plt.savefig('xor_loss5.png', bbox_inches='tight')
plt.show()

