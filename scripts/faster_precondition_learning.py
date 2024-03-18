""" Count and plot the # times this pattern is seen: action is in-plan, no effects seen, the operators changed.

Make a table with a row for each approach, a column for each seed, and a column at the end reporting mean and std.

In a separate function, make individual success plots with dots for each seed.

Baking:

Method                                                                   Mean      Std
----------------------------  --  --  --  --  --  --  --  --  --  --  -------  -------
LLMWarmStart+LNDR // GLIB_G1  20  13  21  15  22  20  17  26  26      20       4.21637
LNDR // GLIB_G1               20   5  14  17  26  10  14  13  20      15.4444  5.81399
LLMWarmStart+LNDR // GLIB_L2  15  11  18  19  19  13  11  15  15  13  14.9     2.84429
LNDR // GLIB_L2               12  18  14  11  10   9  13  13          12.5     2.59808
"""

import matplotlib.pyplot as plt
import os
import pickle
from pprint import pprint
import numpy as np
import gym, pddlgym
from collections import defaultdict
from tabulate import tabulate

LNDR_PATH = '/home/catalan/GLIB-Baking-Fails-and-LLMs/results_openstack/results/LNDR'
GLIB_PATH = '/home/catalan/GLIB-Baking-Fails-and-LLMs/results_openstack/results/GLIB'
RESULTS_PATH = '/home/catalan/GLIB-Baking-Fails-and-LLMs/results_openstack/results'
PLOTS_PATH =  '/home/catalan/GLIB-Baking-Fails-and-LLMs/individual_plots'



def get_dot_plot_counts(domain_name, learning_name, curiosity, seed):
    path = os.path.join(LNDR_PATH, domain_name, curiosity, seed)
    with open(os.path.join(GLIB_PATH, domain_name, learning_name, curiosity, f'{seed}_babbling_stats.pkl'), 'rb') as f:
        babbling_stats = pickle.load(f)
    with open(os.path.join(path, 'skill_sequence.pkl'), 'rb') as f:
        skill_seq = pickle.load(f)
    op_change_itrs = np.loadtxt(os.path.join(path, 'ops_change_iters.txt'))
    last_tdata_dir = -1
    for dir in os.listdir(path):
        if os.path.exists(os.path.join(path, dir, 'transition_data.pkl')) and last_tdata_dir < int(dir[5:]):
            last_tdata_dir = int(dir[5:])
    assert last_tdata_dir != -1
    with open(os.path.join(path, f'iter_{last_tdata_dir}', 'transition_data.pkl'), 'rb') as f:
        t_data = pickle.load(f)

    t_data_idx = defaultdict(lambda: 0)
    counts = []
    for itr, act in enumerate(skill_seq):
        if (t_data_idx[act.predicate]) == (len(t_data[act.predicate])):
            continue
        s,a,e = t_data[act.predicate][t_data_idx[act.predicate]]
        if babbling_stats[itr] not in ("babbled", 'fallback') and (itr in op_change_itrs) and (len(e) == 0):
            counts.append(1)
        else:
            counts.append(0)

        t_data_idx[act.predicate] += 1
    return counts

def individual_dot_plot(counts, domain_name, curiosity, learning_name, seed):
    with open(os.path.join(RESULTS_PATH, domain_name, learning_name, curiosity, f'{domain_name}_{learning_name}_{curiosity}_{seed}.pkl'), 'rb') as f:
        results = pickle.load(f)
    results = np.array(results)
    xs = results[:, 0]
    ys = results[:, 1]
    plt.figure()
    plt.plot(xs, ys, color='#000000')

    y_scat = []
    x_scat = []
    for i, cnt in enumerate(counts):
        if cnt > 0:
            y_scat.append(ys[i])
            x_scat.append(i)
    plt.scatter(x_scat, y_scat,  color='#008000')
    plt.gcf().set_size_inches(22, 14)
    save_path = os.path.join(PLOTS_PATH, domain_name, learning_name, curiosity, 'faster_precond_learning')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{seed}.png'), dpi=300)
    plt.close()
    

domain_name = 'Baking'
datasets = [('LLMWarmStart+LNDR',"GLIB_G1", [str(s) for s in range(120, 130) if s != 125]) , ("LNDR", "GLIB_G1", [str(s) for s in range(100, 110) if s != 106]), ("LLMWarmStart+LNDR", "GLIB_L2", [str(s) for s in range(120, 130)]), ("LNDR", "GLIB_L2", [str(s) for s in range(100, 110) if s not in (108, 107)])]
 
table = []
for learning_name, curiosity, seeds in datasets:
    row = []
    for seed in seeds:
        counts = get_dot_plot_counts(domain_name, learning_name, curiosity, seed)
        individual_dot_plot(counts, domain_name, curiosity, learning_name, seed)
        row.append(sum(counts))
    mean = np.array(row).mean()
    std = np.array(row).std()
    if len(seeds) != 10:
        for _ in range(10 - len(seeds)):
            row.append('')
    row.append(mean)
    row.append(std)
    row = [f'{learning_name} // {curiosity}'] + row
    table.append(row)
headers = ['Method'] + ['' for _ in range(10)] + ['Mean', 'Std']
print(tabulate(table, headers=headers))