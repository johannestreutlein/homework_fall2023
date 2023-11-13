import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_tensorboard_data(logdir, tag='eval_return'):
    ea = EventAccumulator(logdir)
    ea.Reload()
    return [(x.step, x.value) for x in ea.Scalars(tag)]

def plot_data(exp_folders, legends, title, save_path, data_dir):
    fig, ax = plt.subplots()
    for folder, legend in zip(exp_folders, legends):
        eval_data = load_tensorboard_data(os.path.join(data_dir, folder), 'eval_return')
        env_steps_data = load_tensorboard_data(os.path.join(data_dir, folder), 'total_envsteps')
        eval_steps, eval_values = zip(*eval_data)
        env_steps_steps, env_steps_values = zip(*env_steps_data)
        ax.plot(env_steps_values, eval_values, label=legend)
    ax.set_title(title)
    ax.set_xlabel('Env Steps')
    ax.set_ylabel('Eval Return')
    ax.set_ylim(-425, -225)
    ax.grid(True)
    ax.legend()
    plt.savefig(save_path)

def main():
    data_dir = 'data'
    experiments = [
        {
            'folders': [
                'reacher-cs285-v0_reacher_ablation_l2_h250_mpcrandom_horizon10_actionseq1000_29-10-2023_15-20-26',
                'reacher-cs285-v0_reacher_ablation_l2_h250_mpcrandom_horizon10_actionseq1000_27-10-2023_17-35-20',
                'reacher-cs285-v0_reacher_ablation_l2_h250_mpcrandom_horizon10_actionseq1000_29-10-2023_18-55-12',
                #'reacher-cs285-v0_reacher_ablation_l2_h250_mpcrandom_horizon10_actionseq1000_29-10-2023_15-21-31',
            ],
            'legends': ['Ensemble Size 1', 'Ensemble Size 3', 'Ensemble Size 10'],
            'title': 'Effect of Ensemble Size',
            'save_path': os.path.join(data_dir, 'ensemble_size.png'),
        },
        {
            'folders': [
                'reacher-cs285-v0_reacher_ablation_l2_h250_mpcrandom_horizon10_actionseq500_28-10-2023_00-13-54',
                'reacher-cs285-v0_reacher_ablation_l2_h250_mpcrandom_horizon10_actionseq1000_27-10-2023_17-35-20',
                'reacher-cs285-v0_reacher_ablation_l2_h250_mpcrandom_horizon10_actionseq2000_28-10-2023_00-14-27',

            ],
            'legends': ['500 Action Seq', '1000 Action Seq', '2000 Action Seq'],
            'title': 'Effect of Number of Candidate Action Sequences',
            'save_path': os.path.join(data_dir, 'action_sequences.png'),
        },
        {
            'folders': [
                'reacher-cs285-v0_reacher_ablation_l2_h250_mpcrandom_horizon5_actionseq1000_28-10-2023_00-08-20',
                'reacher-cs285-v0_reacher_ablation_l2_h250_mpcrandom_horizon10_actionseq1000_27-10-2023_17-35-20',
                'reacher-cs285-v0_reacher_ablation_l2_h250_mpcrandom_horizon20_actionseq1000_28-10-2023_00-09-53',
            ],
            'legends': ['Horizon 5', 'Horizon 10', 'Horizon 20'],
            'title': 'Effect of Planning Horizon',
            'save_path': os.path.join(data_dir, 'planning_horizon.png'),
        },
    ]
    for exp in experiments:
        plot_data(exp['folders'], exp['legends'], exp['title'], exp['save_path'], data_dir)

if __name__ == '__main__':
    main()
