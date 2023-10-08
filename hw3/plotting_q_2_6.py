import os
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

base_dir = "data/"

exp1 = "hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_08-10-2023_11-58-58"
exp2 = "hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_03-10-2023_17-13-01"
exp3 = "hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_08-10-2023_11-58-23"
exp4 = "hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_08-10-2023_11-57-50"


def extract_data_from_event(event_file):
    env_steps = []
    avg_return = []

    for summary in summary_iterator(event_file):
        for value in summary.summary.value:
            if value.tag == 'eval_return':
                avg_return.append(value.simple_value)
                env_steps.append(summary.step)
            elif value.tag == 'lr':
                lr = value.simple_value


    return env_steps, avg_return, lr

def plot_learning_curve(experiment_sets,  filename, title=None):
    plt.figure(figsize=(8, 5))

    for i, experiment in enumerate(experiment_sets):
        event_file = os.path.join(base_dir, experiment,
                                  [f for f in os.listdir(os.path.join(base_dir, experiment)) if
                                   f.startswith("events.out.tfevents")][0])
        steps, avg_return, lr = extract_data_from_event(event_file)
        if not avg_return:
            continue

        label = f"LR {lr:.1e}"
        plt.plot(steps, avg_return, label=label)

    plt.xlabel('Environment Steps')
    plt.ylabel('Eval Return')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(base_dir, filename), format='pdf')
    plt.close()

# Calling plot_learning_curve with adjusted paths, labels, filename, and title
plot_learning_curve([exp1, exp2, exp3, exp4],
                    "hyperparam_tuning.pdf",
                    title="Vanilla DQN with different learning rates on CartPole-v1")
