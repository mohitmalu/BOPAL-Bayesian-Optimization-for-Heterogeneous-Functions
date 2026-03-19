import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# =============== Plotting Functions for Modeling Error ================= #

def plot_rewards_se(rewards_desc_list, iterations, y_opt, runs, color_list, label_list,
                    title, y_label, d, p, k, N, file_name, ylim=None, shape=False):
    fig, ax = plt.subplots()
    shape_list = ['o', 'v', 's', '*', 'd', 'p', 'x', '+', '>', '<', '1']
    for i in range(len(rewards_desc_list)):
        if shape:
            ax.plot(range(iterations), rewards_desc_list[i].loc['mean'][:iterations], '-', color=color_list[i],
                    label=label_list[i], marker=shape_list[i])
        else:
            ax.plot(range(iterations), rewards_desc_list[i].loc['mean'], '-', color=color_list[i],
                    label=label_list[i])
        lower = rewards_desc_list[i].loc['mean'] - rewards_desc_list[i].loc['std'] / np.sqrt(runs)
        upper = rewards_desc_list[i].loc['mean'] + rewards_desc_list[i].loc['std'] / np.sqrt(runs)
        ax.fill_between(range(iterations), lower, upper, alpha=0.1, color=color_list[i])

    title = title + ", D = " + str(d) + ", Par = " + str(p)
    # ax.set_title('Regret vs Iterations,  Objective optimum = %f' % y_opt)
    # ax.set_ylabel('Regret')
    # ax.set_xlabel('Iterations')
    # ax.set_ylim((-0.01,0.99)) ## Setting the limit to have consistent Y range representation
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_ylim(ylim)
    ax.set_xlabel('Iterations')
    plt.legend()
    fig.tight_layout()
    img_file_name = file_name + ".png"
    plt.savefig(img_file_name)
    plt.show()


#     print("Reward 1 maximum mean = ", max(rewards_desc1.loc['mean']),
#           " and Reward 2 maximum mean = ", max(rewards_desc2.loc['mean']))



