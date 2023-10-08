import os
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

base_dir = "data/"

single_q = "hw3_sac_sac_hopper_singlecritic_Hopper-v4_reparametrize_s128_l3_alr0.0003_clr0.0003_b256_d0.99_t0.05_stu0.005_06-10-2023_17-49-12"
double_q = "hw3_sac_sac_hopper_doubleq_Hopper-v4_reparametrize_s128_l3_alr0.0003_clr0.0003_b256_d0.99_t0.05_stu0.005_doubleq_06-10-2023_17-48-54"
clipped_q = "hw3_sac_sac_hopper_clipq_Hopper-v4_reparametrize_s128_l3_alr0.0003_clr0.0003_b256_d0.99_t0.05_stu0.005_min_06-10-2023_17-48-09"

def get_legend_label(experiment):
    if "singlecritic" in experiment:
        return "Single Critic"
    elif "doubleq" in experiment:
        return "Double Q"
    elif "clipq" in experiment:
        return "Clipped Q"
    return ""

def extract_data_from_event(event_file, tag):
    env_steps = []
    avg_return = []

    for summary in summary_iterator(event_file):
        for value in summary.summary.value:
            if value.tag == tag:
                avg_return.append(value.simple_value)
                env_steps.append(summary.step)

    return env_steps, avg_return

def plot_learning_curve(experiment_sets,  filename, title=None):
    plt.figure(figsize=(8, 5))

    for i, experiment in enumerate(experiment_sets):
        event_file = os.path.join(base_dir, experiment,
                                  [f for f in os.listdir(os.path.join(base_dir, experiment)) if
                                   f.startswith("events.out.tfevents")][0])
        steps, avg_return = extract_data_from_event(event_file, 'eval_return')
        if not avg_return:
            continue

        label = get_legend_label(experiment) + ", Eval Return"
        plt.plot(steps, avg_return, label=label, alpha=0.7)

    for i, experiment in enumerate(experiment_sets):
        event_file = os.path.join(base_dir, experiment,
                                  [f for f in os.listdir(os.path.join(base_dir, experiment)) if
                                   f.startswith("events.out.tfevents")][0])
        steps, avg_return = extract_data_from_event(event_file, 'q_values')
        if not avg_return:
            continue

        label = get_legend_label(experiment) + ", Q Values"
        plt.plot(steps, avg_return, label=label, alpha=0.7)

    plt.xlabel('Environment Steps')
    plt.ylabel('')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(base_dir, filename), format='pdf')
    plt.close()

# Calling plot_learning_curve with adjusted paths, labels, filename, and title
plot_learning_curve([single_q, double_q, clipped_q],
                    "stabilizing_target_values.pdf",
                    title="SAC on Hopper-v4",)

