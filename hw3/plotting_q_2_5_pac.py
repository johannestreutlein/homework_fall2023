import os
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

base_dir = "data/"

result_folder = "hw3_dqn_dqn_MsPacmanNoFrameskip-v0_d0.99_tu2000_lr0.0001_doubleq_clip10.0_04-10-2023_20-26-06"


def extract_data_from_event(event_file):
    eval_steps = []
    eval_return = []
    train_steps = []
    train_return = []

    for summary in summary_iterator(event_file):
        for value in summary.summary.value:
            if value.tag == 'eval_return':
                eval_return.append(value.simple_value)
                eval_steps.append(summary.step)
            if value.tag == 'train_return':
                train_return.append(value.simple_value)
                train_steps.append(summary.step)

    return eval_steps, eval_return, train_steps, train_return

def plot_learning_curve(experiment_sets,  filename, title=None):
    plt.figure(figsize=(8, 5))

    for i, experiment in enumerate(experiment_sets):
        label_added = False
        event_file = os.path.join(base_dir, experiment,
                                  [f for f in os.listdir(os.path.join(base_dir, experiment)) if
                                   f.startswith("events.out.tfevents")][0])
        eval_steps, eval_return, train_steps, train_return = extract_data_from_event(event_file)



        label = "Train Return"
        plt.plot(train_steps, train_return, label=label)

        label = "Eval Return"
        plt.plot(eval_steps, eval_return, label=label)

    plt.xlabel('Environment Steps')
    plt.ylabel('Eval Return')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(base_dir, filename), format='pdf')
    plt.close()

# Calling plot_learning_curve with adjusted paths, labels, filename, and title
plot_learning_curve([result_folder], "dqn_pacman.pdf",
                    title="Double DQN on MsPacmanNoFrameskip-v0")
