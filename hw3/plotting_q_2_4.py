import os
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

base_dir = "data/"

experiment_folders = [folder for folder in os.listdir(base_dir) if folder.startswith("hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_03-10-2023_16-52-23")]

def extract_data_from_event(event_file):
    env_steps = []
    avg_return = []

    for summary in summary_iterator(event_file):
        for value in summary.summary.value:
            if value.tag == 'eval_return':
                avg_return.append(value.simple_value)
                env_steps.append(summary.step)

    return env_steps, avg_return


def get_legend_label(experiment):
    if "rtg" in experiment and "na" in experiment:
        return "RTG + NA"
    elif "rtg" in experiment:
        return "RTG"
    elif "na" in experiment:
        return "NA"
    else:
        return ""


def plot_learning_curve(experiments, filename, title=None):
    plt.figure(figsize=(8, 5))

    for experiment in experiments:
        event_file = os.path.join(base_dir, experiment, [f for f in os.listdir(os.path.join(base_dir, experiment)) if
                                                         f.startswith("events.out.tfevents")][0])

        steps, avg_return = extract_data_from_event(event_file)

        if not avg_return:
            continue

        label = get_legend_label(experiment)
        plt.plot(steps, avg_return, label=label)

    plt.xlabel('Number of Environment Steps')
    plt.ylabel('Eval Return')
    plt.title(title)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, filename), format='pdf')
    plt.close()


plot_learning_curve(experiment_folders,
                    "dqn_cartpole.pdf", title="DQN on CartPole-v1")
