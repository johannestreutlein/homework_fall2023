import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_tensorboard_data(logdir, tag='eval_return'):
    ea = EventAccumulator(logdir)
    ea.Reload()
    return [x.value for x in ea.Scalars(tag)]

def plot_data(eval_data, env_steps_data, env_name, save_path):
    eval_values = eval_data
    env_steps_values = env_steps_data
    print(env_steps_values)
    print(eval_values)

    fig, ax = plt.subplots()
    ax.plot(env_steps_values, eval_values)
    ax.set_title(env_name)
    ax.set_xlabel('Env Steps')
    ax.set_ylabel('Eval Return')
    plt.savefig(save_path)

def main():
    data_dir = 'data'
    folders = [
        ('obstacles-cs285-v0_obstacles_multi_l2_h250_mpcrandom_horizon10_actionseq1000_27-10-2023_16-42-31', 'Obstacles Eval Return'),
        ('reacher-cs285-v0_reacher_multi_l2_h250_mpcrandom_horizon10_actionseq1000_27-10-2023_16-52-22', 'Reacher Eval Return'),
        ('cheetah-cs285-v0_cheetah_multi_l2_h250_mpcrandom_horizon15_actionseq1000_27-10-2023_16-52-40', 'Cheetah Eval Return'),
    ]
    for folder, env_name in folders:
        logdir = os.path.join(data_dir, folder)
        eval_data = load_tensorboard_data(logdir, 'eval_return')
        env_steps_data = load_tensorboard_data(logdir, 'total_envsteps')
        save_path = os.path.join(data_dir, f"{folder}.png")
        plot_data(eval_data, env_steps_data, env_name, save_path)

if __name__ == '__main__':
    main()
