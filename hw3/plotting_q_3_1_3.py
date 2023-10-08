import os
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

base_dir = "data/"

reinforce1_folder = "hw3_sac_reinforce1_HalfCheetah-v4_reinforce_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.2_stu0.005_06-10-2023_16-57-36"
reinforce10_folder = "hw3_sac_reinforce10_HalfCheetah-v4_reinforce_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.2_stu0.005_06-10-2023_16-59-53"
reparam_folder = "hw3_sac_reparametrize_HalfCheetah-v4_reparametrize_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.1_stu0.005_06-10-2023_17-14-36"

def get_legend_label(experiment):
    if "reinforce10" in experiment:
        return "REINFORCE-10"
    elif "reinforce1" in experiment:
        return "REINFORCE-1"
    elif "reparametrize" in experiment:
        return "Reparametrize"
    return ""

def extract_data_from_event(event_file):
    env_steps = []
    avg_return = []

    for summary in summary_iterator(event_file):
        for value in summary.summary.value:
            if value.tag == 'eval_return':
                avg_return.append(value.simple_value)
                env_steps.append(summary.step)

    return env_steps, avg_return

def plot_learning_curve(experiment_sets,  filename, title=None):
    plt.figure(figsize=(8, 5))

    for i, experiment in enumerate(experiment_sets):
        label_added = False
        event_file = os.path.join(base_dir, experiment,
                                  [f for f in os.listdir(os.path.join(base_dir, experiment)) if
                                   f.startswith("events.out.tfevents")][0])
        steps, avg_return = extract_data_from_event(event_file)
        if not avg_return:
            continue

        label = get_legend_label(experiment) if not label_added else ""
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
plot_learning_curve([reinforce1_folder, reinforce10_folder, reparam_folder],
                    "reinforce_comparison_halfcheetah.pdf",
                    title="REINFORCE-1 vs REINFORCE-10 vs Reparametrize on HalfCheetah-v4")
