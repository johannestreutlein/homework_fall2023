import os
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

base_dir = "data/"

doubleq_folders = [folder for folder in os.listdir(base_dir) if
                   folder.startswith("hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_doubleq")]
vanilla_folders = [folder for folder in os.listdir(base_dir) if
                   folder.startswith("hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99") and "doubleq" not in folder]


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
    return "Double Q" if "doubleq" in experiment else "Vanilla DQN"


def plot_learning_curve(experiment_sets, colors, filename, title=None):
    plt.figure(figsize=(8, 5))

    for i, experiments in enumerate(experiment_sets):
        label_added = False
        for experiment in experiments:
            event_file = os.path.join(base_dir, experiment,
                                      [f for f in os.listdir(os.path.join(base_dir, experiment)) if
                                       f.startswith("events.out.tfevents")][0])
            steps, avg_return = extract_data_from_event(event_file)
            if not avg_return:
                continue

            label = get_legend_label(experiment) if not label_added else ""
            plt.plot(steps, avg_return, color=colors[i], label=label)
            label_added = True

    plt.xlabel('Environment Steps')
    plt.ylabel('Eval Return')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(base_dir, filename), format='pdf')
    plt.close()


plot_learning_curve([doubleq_folders, vanilla_folders], ['r', 'b'], "dqn_lunarlander_doubleq_vs_vanilla.pdf",
                    title="Double Q vs Vanilla DQN on LunarLander-v2")
