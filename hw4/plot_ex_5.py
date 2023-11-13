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
    ax.legend()
    plt.savefig(save_path)

def main():
    data_dir = 'data'
    experiments = [
        {
            'folders': [
                'cheetah-cs285-v0_cheetah_multi_l2_h250_mpcrandom_horizon15_actionseq1000_27-10-2023_16-52-40',
                'cheetah-cs285-v0_cheetah_cem_l2_h250_mpccem_horizon15_actionseq1000_cem_iters4_27-10-2023_18-16-10',
                'cheetah-cs285-v0_cheetah_cem_l2_h250_mpccem_horizon15_actionseq1000_cem_iters2_28-10-2023_00-35-44',
            ],
            'legends': ['Random Shooting', 'Cross Entropy Method (4 iters)', 'Cross Entropy Method (2 iters)'],
            'title': 'Random Shooting vs Cross Entropy Method (Cheetah)',
            'save_path': os.path.join(data_dir, 'CEM_v_random_shooting.png'),
        },
    ]
    for exp in experiments:
        plot_data(exp['folders'], exp['legends'], exp['title'], exp['save_path'], data_dir)

if __name__ == '__main__':
    main()
