import tensorflow as tf
import tensorflow_datasets as tfds
import numpy
from keras.callbacks import EarlyStopping, ModelCheckpoint
import random
import math
from pathlib import Path


def initialize_dataset(ds, info):
    num_labels = info.features['label'].num_classes
    features = []
    labels = []

    for example in ds:  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`
        features.append(example["features"].numpy().tolist())
        labels.append(example["label"].numpy().tolist())

    for index, label in enumerate(labels):
        nn_label = numpy.zeros(num_labels, dtype=int)
        nn_label[label] = 1
        labels[index] = nn_label.tolist()

    return features, labels


def get_weight_count(weights):
    count = 0
    for index, _list in enumerate(weights):
        for j, sub_list in enumerate(_list):
            try:
                len(sub_list)
                count += len(sub_list)
            except:
                count += 1
    print("count", count)
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
        weights = numpy.array(nn_model.get_weights())
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
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(units=4, activation='relu'),
        tf.keras.layers.Dense(units=3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor='loss', patience=20),
                 ModelCheckpoint(filepath='best_weights.h5', monitor='loss', save_weights_only=True,
                                 save_best_only=True)]
    model.fit(
        x=train_features,
        y=train_labels,
        epochs=2000,
        steps_per_epoch=15,
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

    weight_array = numpy.array(model.get_weights())
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
    run_amount = 2
    run_num = 8
    weight_amount = 0
    weight_modification_step_amount = 10

    (train, test), ds_info = tfds.load('iris', split=['train[:70%]', 'train[30%:]'], shuffle_files=True,
                                       with_info=True, )
    train_features, train_labels = initialize_dataset(train, ds_info)
    test_features, test_labels = initialize_dataset(test, ds_info)

    for i in range(run_start, run_start + run_amount):
        print()
        print(
            "==========================================================================================================")
        print(f"Experiment {i} of {run_start + run_amount -1}")
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
        numpy.savetxt(f'Experiments/Data_Collection/Run_{run_num}/Train_Vals/train_left_vals{i}.csv', train_left,
                      delimiter=',')
        numpy.savetxt(f'Experiments/Data_Collection/Run_{run_num}/Train_Vals/train_optima{i}.csv', numpy.array(train_evals),
                      delimiter=',')
        numpy.savetxt(f'Experiments/Data_Collection/Run_{run_num}/Train_Vals/train_right_vals{i}.csv', train_right,
                      delimiter=',')

        test_vals_dir = Path.cwd() / "Experiments" / "Data_Collection" / f"Run_{run_num}" / "Test_Vals"
        if not Path.exists(test_vals_dir):
            test_vals_dir.mkdir(parents=True, exist_ok=True)
        numpy.savetxt(f'Experiments/Data_Collection/Run_{run_num}/Test_Vals/test_left_vals{i}.csv', test_left,
                      delimiter=',')
        numpy.savetxt(f'Experiments/Data_Collection/Run_{run_num}/Test_Vals/test_optima{i}.csv', numpy.array(test_evals),
                      delimiter=',')
        numpy.savetxt(f'Experiments/Data_Collection/Run_{run_num}/Test_Vals/test_right_vals{i}.csv', test_right,
                      delimiter=',')
