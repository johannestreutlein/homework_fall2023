import os
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

base_dir = "data/"

experiment_folders = ["hw3_sac_sac_humanoid_Humanoid-v4_reparametrize_s256_l3_alr0.0003_clr0.0003_b256_d0.99_t0.05_stu0.005_min_06-10-2023_20-11-01"]

def extract_data_from_event(event_file):
    env_steps = []
    avg_return = []

    for summary in summary_iterator(event_file):
        for value in summary.summary.value:
            if value.tag == 'eval_return':
                avg_return.append(value.simple_value)
                env_steps.append(summary.step)

    return env_steps, avg_return




def plot_learning_curve(experiments, filename, title=None):
    plt.figure(figsize=(8, 5))

    for experiment in experiments:
        event_file = os.path.join(base_dir, experiment, [f for f in os.listdir(os.path.join(base_dir, experiment)) if
                                                         f.startswith("events.out.tfevents")][0])

        steps, avg_return = extract_data_from_event(event_file)

        if not avg_return:
            continue

        plt.plot(steps, avg_return)

    plt.xlabel('Environment Steps')
    plt.ylabel('Eval Return')
    plt.title(title)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, filename), format='pdf')
    plt.close()


plot_learning_curve(experiment_folders,
                    "sac_humanoid.pdf", title="SAC Reparametrize on Humanoid-v4")
