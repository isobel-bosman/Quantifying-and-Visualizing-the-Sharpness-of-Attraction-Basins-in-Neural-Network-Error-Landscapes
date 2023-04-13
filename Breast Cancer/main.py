import numpy as np
# import numpy
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
import random
import math
from pathlib import Path


def initialize_dataset():
    data = pd.read_csv("data.csv")

    features = data.iloc[:, 2:32]

    data['diagnosis'][data['diagnosis'] == 'M'] = 0
    data['diagnosis'][data['diagnosis'] == 'B'] = 1
    labels = data['diagnosis']

    train_f, test_f, train_l, test_l = train_test_split(features, labels, random_state=42)

    scaler = StandardScaler()
    train_f = scaler.fit_transform(train_f)
    test_f = scaler.transform(test_f)

    return train_f, np.asarray(train_l).astype('float32'), test_f, np.asarray(test_l).astype('float32')


# Count all the entities in the weight array
def get_weight_count(weights):
    count = 0
    for index, _list in enumerate(weights):
        for j, sub_list in enumerate(_list):
            try:
                len(sub_list)
                count += len(sub_list)
            except:
                count += 1
    return count


# Find indices of half of the count
def get_index_subset(c):
    indices = list(range(c))
    if (c % 2) == 0:
        half = c / 2
    else:
        addition = random.randint(0, 1)
        half = math.floor(c / 2)
        half += addition

    subset = random.sample(indices, half)
    return subset


# modify the weights
def modify_weights(weights, modification_amount, count):
    # Get random indexes of half of the weights
    indices = get_index_subset(count)

    # Modify weights at the chosen indexes
    index = -1
    for i, weight_list in enumerate(weights):
        for j, sub_list in enumerate(weight_list):
            try:
                len(sub_list)
                for k, sub_sub_list in enumerate(sub_list):
                    index += 1
                    if index in indices:
                        # print("chosen", index)
                        weights[i][j][k] += modification_amount
                    # print('a[{}][{}][{}] = {}'.format(i, j, k, sub_sub_list))
            except:
                index += 1
                if index in indices:
                    # print("chosen", index)
                    weights[i][j] += modification_amount
                # print('a[{}][{}] = {}'.format(i, j, sub_list))
    return weights


def calculate_modified_loss_array(modification_amount, loss_amount, weight_count):
    train_loss_array = []
    test_loss_array = []
    nn_model = tf.keras.models.load_model('trained_model.h5')

    for index in range(loss_amount):
        nn_model.load_weights('best_weights.h5')
        weights = np.array(nn_model.get_weights())
        mod_weights = modify_weights(weights, modification_amount, weight_count)
        nn_model.set_weights(mod_weights)

        # evaluate with training data
        eval_model = nn_model.evaluate(train_features, train_labels, verbose=0)
        modified_loss = eval_model[0]
        train_loss_array.append(modified_loss)

        # evaluate with testing data
        eval_model = nn_model.evaluate(test_features, test_labels, verbose=0)
        modified_loss = eval_model[0]
        test_loss_array.append(modified_loss)

    return train_loss_array, test_loss_array

def create_model():
    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(30,)),
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor='loss', patience=10),
                 ModelCheckpoint(filepath='best_weights.h5', monitor='loss', save_weights_only=True,
                                 save_best_only=True)]

    model.fit(
        x=train_features,
        y=train_labels,
        epochs=100,
        steps_per_epoch=10,
        callbacks=callbacks,
        verbose=0)

    # Save the model after convergence
    model.save('trained_model.h5')

def run_experiment(modification_step_amount, weight_count):
    # Find the loss value after convergence
    model = tf.keras.models.load_model('trained_model.h5')

    # Find the loss value after convergence
    model.load_weights('best_weights.h5')
    eval_model = model.evaluate(train_features, train_labels)
    train_eval = eval_model

    eval_model = model.evaluate(test_features, test_labels)
    test_eval = eval_model

    weight_array = np.array(model.get_weights())
    if weight_count == 0:
        weight_count = get_weight_count(weight_array)

    # Generate Losses for Boxplot
    subtracted_results = []
    addition_results = []
    subtracted_test_results = []
    addition_test_results = []

    step = 0.01
    current_step = (modification_step_amount * step) * -1

    for modification_step in range(modification_step_amount * 2):
        train_losses, test_losses = calculate_modified_loss_array(current_step, 30, weight_count)
        if current_step > 0:
            addition_results.append(train_losses)
            addition_test_results.append(test_losses)
        if current_step < 0:
            subtracted_results.append(train_losses)
            subtracted_test_results.append(test_losses)
        current_step += step
        current_step = round(current_step, 2)
        if current_step == 0:
            current_step += step
            current_step = round(current_step, 2)

    return subtracted_results, train_eval, addition_results, subtracted_test_results, test_eval, addition_test_results, weight_count


# Main run
if __name__ == '__main__':
    run_start = 1
    run_amount = 1
    run_num = 1
    weight_modification_step_amount = 10
    weight_amount = 0

    train_features, train_labels, test_features, test_labels = initialize_dataset()

    for i in range(run_start, run_start + run_amount):
        print()
        print(
            "==========================================================================================================")
        print(f"Experiment {i} of {run_start + run_amount - 1}")
        print(
            "==========================================================================================================")
        print()

        # run experiment
        create_model()
        train_left, train_evals, train_right, test_left, test_evals, test_right, weight_amount = run_experiment(
            weight_modification_step_amount, weight_amount)

        train_vals_dir = Path.cwd() / "Experiments" / "Data_Collection" / f"Run_{run_num}" / "Train_Vals"
        if not Path.exists(train_vals_dir):
            train_vals_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(f'Experiments/Data_Collection/Run_{run_num}/Train_Vals/train_left_vals{i}.csv', train_left,
                      delimiter=',')
        np.savetxt(f'Experiments/Data_Collection/Run_{run_num}/Train_Vals/train_optima{i}.csv',
                      np.array(train_evals),
                      delimiter=',')
        np.savetxt(f'Experiments/Data_Collection/Run_{run_num}/Train_Vals/train_right_vals{i}.csv', train_right,
                      delimiter=',')

        test_vals_dir = Path.cwd() / "Experiments" / "Data_Collection" / f"Run_{run_num}" / "Test_Vals"
        if not Path.exists(test_vals_dir):
            test_vals_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(f'Experiments/Data_Collection/Run_{run_num}/Test_Vals/test_left_vals{i}.csv', test_left,
                      delimiter=',')
        np.savetxt(f'Experiments/Data_Collection/Run_{run_num}/Test_Vals/test_optima{i}.csv',
                      np.array(test_evals),
                      delimiter=',')
        np.savetxt(f'Experiments/Data_Collection/Run_{run_num}/Test_Vals/test_right_vals{i}.csv', test_right,
                      delimiter=',')