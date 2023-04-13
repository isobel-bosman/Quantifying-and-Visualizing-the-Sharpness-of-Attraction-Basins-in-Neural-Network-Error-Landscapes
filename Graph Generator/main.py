import numpy
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from numpy import genfromtxt
matplotlib.use('TkAgg')


def generate_x_tick_values(modification_step_amount):
    x_tick_values = []
    step = 0.01
    current_step = (modification_step_amount * step) * -1

    for modification_step in range(modification_step_amount * 2):
        x_tick_values.append(f"{'+' if (current_step > 0) else ''}{str(current_step)}")
        current_step += step
        current_step = round(current_step, 2)
        if current_step == 0:
            x_tick_values.append("0")
            current_step += step
            current_step = round(current_step, 2)

    return x_tick_values


def generate_fitness_landscape_approximation(left_plot, right_plot, x_tick_vals, x_tick_labels, y_max, path):
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    fig.set_size_inches(20, 15)
    y_ticks = numpy.arange(0.0, y_max + 0.1, (0.1 if y_max <= 10.0 else 0.2))

    ax1.boxplot(left_plot)
    ax1.set_title(f"Fitness landscape approximation around local optima from a Neural Network trained\n"
                  f"on the {dataset_name} dataset: Loss values determined by training dataset")
    ax1.set_xticks(x_tick_vals, x_tick_labels, rotation=30)
    ax1.set_xlabel('Modification amount for a random sample of half the weights')
    ax1.set_yticks(y_ticks)
    ax1.set_ylabel('Loss Value')
    ax1.set_ylim([0.0, y_max + 0.1])

    ax2.boxplot(right_plot)
    ax2.set_title(f"Fitness landscape approximation around local optima from a Neural Network trained\n"
                  f"on the {dataset_name} dataset: Loss values determined by testing dataset")
    ax2.set_xticks(x_tick_vals, x_tick_labels, rotation=30)
    ax2.set_xlabel('Modification amount for a random sample of half the weights')
    ax2.set_yticks(y_ticks)
    ax2.set_ylabel('Loss Value')
    ax1.set_ylim([0.0, y_max + 0.1])

    plt.savefig(path, bbox_inches='tight', dpi=100)
    plt.close()


def generate_sharpness_correlation_scatter_plot(sharpness_vals, generalisation_vals, symmetry_vals, path):
    plt.scatter(sharpness_vals, generalisation_vals, c=symmetry_vals, cmap='plasma', vmin=0, vmax=1)
    cbar = plt.colorbar()
    cbar.set_label('Highly asymmetric = 1\nHighly symmetric = 0\n', rotation=270, labelpad=30)
    plt.xlabel("Sharpness approximation value around a local optima")
    plt.ylabel("Difference between testing loss and training loss")
    plt.title(
        "Correlation between the sharpness approximation value around\n a local optima of a trained NN and its ability"
        " to generalise")
    plt.savefig(path, bbox_inches='tight', dpi=100)
    plt.close()


if __name__ == '__main__':
    run_start = 1
    run_amount = 100
    run_num = 1
    dataset_name = 'Mnist'
    # path_to_experiments_run_folder = r"C:\Users\isobe\PycharmProjects\Iris\Run_1"
    path_to_experiments_run_folder = r"C:\Users\isobe\PycharmProjects\Mnist\Run_1"
    weight_modification_step_amount = 10

    test_all_vals = []
    train_all_vals = []
    train_optima_vals = []
    test_optima_vals = []
    generalisation_scores = []
    sharpness_scores = []
    symmetry_scores = []

    max_vals = []

    for i in range(1, 1 + run_amount):
        train_vals = []
        test_vals = []
        train_path = r"/Train_Vals/"
        test_path = r"/Test_Vals/"
        train_left = genfromtxt(f'{path_to_experiments_run_folder}{train_path}train_left_vals{run_start + i - 1}.csv',
                                delimiter=',')
        train_optima = genfromtxt(f'{path_to_experiments_run_folder}{train_path}train_optima{run_start + i - 1}.csv',
                                delimiter=',')
        train_right = genfromtxt(f'{path_to_experiments_run_folder}{train_path}train_right_vals{run_start + i - 1}.csv',
                                 delimiter=',')
        test_left = genfromtxt(f'{path_to_experiments_run_folder}{test_path}test_left_vals{run_start + i - 1}.csv',
                                delimiter=',')
        test_optima = genfromtxt(f'{path_to_experiments_run_folder}{test_path}test_optima{run_start + i - 1}.csv',
                                  delimiter=',')
        test_right = genfromtxt(f'{path_to_experiments_run_folder}{test_path}test_right_vals{run_start + i - 1}.csv',
                                 delimiter=',')

        left_max = numpy.max(numpy.max(train_left))
        right_max = numpy.max(numpy.max(train_right))
        max_vals.append(numpy.max([left_max, right_max]))

        sharpness_scores.append(left_max + right_max)
        symmetry_score = (left_max - right_max) / max(left_max, right_max)
        symmetry_scores.append(abs(symmetry_score))
        generalisation_scores.append(test_optima - train_optima)

        for j in range(0, len(train_left)):
            train_vals.append(list(train_left[j]))
            test_vals.append(list(test_left[j]))
        train_vals.append([train_optima])
        test_vals.append([test_optima])
        for j in range(0, len(train_right)):
            train_vals.append(list(train_right[j]))
            test_vals.append(list(test_right[j]))

        train_all_vals.append(train_vals)
        test_all_vals.append(test_vals)
        train_optima_vals.append(train_optima)
        test_optima_vals.append(test_optima)

    max_val = numpy.max(max_vals)

    numpy.savetxt(f'{path_to_experiments_run_folder}/train_optima_vals{i}.csv', train_optima_vals, delimiter=',')
    numpy.savetxt(f'{path_to_experiments_run_folder}/test_optima_vals{i}.csv', test_optima_vals, delimiter=',')

    graph_dir = Path.cwd() / dataset_name / f"Run_{run_num}"
    if not Path.exists(graph_dir):
        graph_dir.mkdir(parents=True, exist_ok=True)

    x_ticks = generate_x_tick_values(weight_modification_step_amount)

    for i in range(0, run_amount):
        generate_fitness_landscape_approximation(left_plot=train_all_vals[i],
                                                 right_plot=test_all_vals[i],
                                                 x_tick_vals=list(range(1, (weight_modification_step_amount * 2) + 2)),
                                                 x_tick_labels=x_ticks,
                                                 y_max=max_val,
                                                 path=f"{graph_dir}/{dataset_name}_landscape_{i}.png")

    generate_sharpness_correlation_scatter_plot(sharpness_vals=sharpness_scores,
                                                generalisation_vals=generalisation_scores,
                                                symmetry_vals=symmetry_scores,
                                                path=f"{graph_dir}/{dataset_name}_correlation_plot.png")

    print("graphs generated in the following directory:", graph_dir)
    print("Sharpest:", numpy.max(sharpness_scores))
    print("Flattest:", numpy.min(sharpness_scores))
