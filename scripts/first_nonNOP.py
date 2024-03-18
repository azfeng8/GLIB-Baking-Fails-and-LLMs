"""Print tables to display information about the first nonNOP.

The first four rows show the iteration number of the first nonNOP.

The second half shows the counts, over 10 seeds, how many times the first nonNOP was the result of a babbled action, a fallback action, or an action in a plan.

Method                        bakecake                 bakesouffle              cleanpan                 mix                      putegginpan              putflourinpan           putpaninoven           removepanfromoven
----------------------------  -----------------------  -----------------------  -----------------------  -----------------------  -----------------------  ----------------------  ---------------------  ---------------------
LLMWarmStart+LNDR // GLIB_G1  327.8889 (std=515.3412)  900.5714 (std=511.3235)  567.6667 (std=460.4983)  90.6667 (std=218.7226)   76.0 (std=214.2543)      2.0 (std=2.1082)        5.6667 (std=3.1269)    54.7778 (std=43.802)
LNDR // GLIB_G1               830.8333 (std=638.1942)  508.8889 (std=439.2361)  391.6667 (std=491.1381)  722.8571 (std=680.6624)  125.4444 (std=318.9779)  431.6667 (std=501.375)  20.3333 (std=21.6795)  36.1111 (std=23.5723)
LLMWarmStart+LNDR // GLIB_L2  312.4444 (std=375.1406)  490.5 (std=413.192)      389.5 (std=459.8539)     95.7 (std=279.1057)      1.7 (std=4.776)          1.4 (std=1.1136)        6.0 (std=6.3561)       17.8 (std=13.4447)
LNDR // GLIB_L2               377.625 (std=336.5163)   548.875 (std=372.4475)   168.875 (std=396.9548)   268.875 (std=334.5151)   25.0 (std=19.666)        35.75 (std=34.2409)     20.625 (std=12.7469)   59.5 (std=32.2219)

LLMWarmStart+LNDR // GLIB_G1
_  babbled:                   1                        5                        9                        2                        1                        0                       2                      1
_  fallback:                  8                        2                        0                        1                        0                        1                       0                      7
_  inplan:                    0                        0                        0                        6                        8                        8                       7                      1
LNDR // GLIB_G1
_  babbled:                   0                        4                        4                        6                        1                        5                       0                      2
_  fallback:                  6                        5                        5                        1                        8                        4                       9                      7
_  inplan:                    0                        0                        0                        0                        0                        0                       0                      0
LLMWarmStart+LNDR // GLIB_L2
_  babbled:                   3                        1                        0                        0                        0                        0                       0                      9
_  fallback:                  6                        9                        10                       1                        1                        1                       1                      1
_  inplan:                    0                        0                        0                        9                        9                        9                       9                      0
LNDR // GLIB_L2
_  babbled:                   1                        1                        0                        6                        0                        3                       0                      5
_  fallback:                  7                        7                        8                        2                        8                        5                       8                      3
_  inplan:                    0                        0                        0                        0                        0                        0                       0                      0
"""

import os
import pickle
from pprint import pprint
import numpy as np
import gym, pddlgym
from collections import defaultdict
from tabulate import tabulate # https://pypi.org/project/tabulate/

LNDR_PATH = '/home/catalan/GLIB-Baking-Fails-and-LLMs/results_openstack/results/LNDR'
GLIB_PATH = '/home/catalan/GLIB-Baking-Fails-and-LLMs/results_openstack/results/GLIB'

def get_data(domain_name, curiosity, learning_name, seeds):
    """Gets data about the first nonNOP for each skill.

    Args:
        domain_name (_type_): _description_
        curiosity (_type_): _description_
        learning_name (_type_): _description_
        seeds (_type_): _description_
    Returns:
        maps from skill to data
    """

    babbled_nonNOP_action = defaultdict(lambda:0)
    fallback_nonNOP_action = defaultdict(lambda:0)
    inplan_nonNOP_action = defaultdict(lambda:0)

    first_nonNOP_iter = defaultdict(list)
    for seed in seeds:
        path = os.path.join(LNDR_PATH, domain_name, curiosity, seed)
        # Find the last iteration with transition data
        last_tdata_dir = -1
        for dir in os.listdir(path):
            if os.path.exists(os.path.join(path, dir, 'transition_data.pkl')) and last_tdata_dir < int(dir[5:]):
                last_tdata_dir = int(dir[5:])
        assert last_tdata_dir != -1
        with open(os.path.join(path, f'iter_{last_tdata_dir}', 'transition_data.pkl'), 'rb') as f:
            t_data = pickle.load(f)

        with open(os.path.join(path, 'skill_sequence.pkl'), 'rb') as f:
            skill_seq = pickle.load(f)

        with open(os.path.join(GLIB_PATH, domain_name, learning_name, curiosity, f'{seed}_babbling_stats.pkl'), 'rb') as f:
            babbling_stats = pickle.load(f)

        # first_nonNOP = defaultdict(lambda : 0)
        for skill in t_data:
            j = 0
            iter_num = 0
            for act in skill_seq:
                if j == len(t_data[skill]):
                    break
                s,a,e = t_data[skill][j]
                if act == a:
                    if len(e) != 0:
                        if babbling_stats[iter_num] == 'fallback':
                            fallback_nonNOP_action[skill] += 1
                        elif babbling_stats[iter_num] == 'babbled':
                            babbled_nonNOP_action[skill] += 1
                        else:
                            inplan_nonNOP_action[skill] += 1
                        first_nonNOP_iter[skill].append(iter_num)
                        # first_nonNOP[skill] = (iter_num, 'in plan')
                        # first_nonNOP[skill] = (iter_num, babbling_stats[iter_num])
                        break
                    j += 1
                iter_num += 1
    return first_nonNOP_iter, fallback_nonNOP_action, babbled_nonNOP_action, inplan_nonNOP_action

domain_name = 'Baking'
env = pddlgym.make(f'PDDLEnv{domain_name}-v0')
skills = [p.name for p in env.action_space.predicates]

datasets = [('LLMWarmStart+LNDR',"GLIB_G1", [str(s) for s in range(120, 130) if s != 125]) , ("LNDR", "GLIB_G1", [str(s) for s in range(100, 110) if s != 106]), ("LLMWarmStart+LNDR", "GLIB_L2", [str(s) for s in range(120, 130)]), ("LNDR", "GLIB_L2", [str(s) for s in range(100, 110) if s not in (108, 107)])]
                   

nonNOP_iter_table = []
nonNOP_action_type_table = []
headers = ['Method'] + skills
for learning_name, curiosity, seeds in datasets:
    first_nonNOP_iter, fallback_nonNOP_action, babbled_nonNOP_action, inplan_nonNOP_action = get_data(domain_name, curiosity, learning_name, seeds)
    iter_table_row = [f'{learning_name} // {curiosity}']
    action_table_row_header = [f'{learning_name} // {curiosity}']
    action_table_row_babble = ['_  babbled:']
    action_table_row_inplan = ['_  inplan:']
    action_table_row_fallback = ['_  fallback:']
    for skill in skills:
        if skill in first_nonNOP_iter:
            arr = np.array(first_nonNOP_iter[skill])
            iter_table_row.append(f'{round(arr.mean(), 4)} (std={round(arr.std(), 4)})')
        else:
            iter_table_row.append('')

        action_table_row_babble.append(babbled_nonNOP_action[skill])
        action_table_row_inplan.append(inplan_nonNOP_action[skill])
        action_table_row_fallback.append(fallback_nonNOP_action[skill])
        action_table_row_header.append('')

    nonNOP_iter_table.append(iter_table_row)
    nonNOP_action_type_table.append(action_table_row_header)
    nonNOP_action_type_table.append(action_table_row_babble)
    nonNOP_action_type_table.append(action_table_row_fallback)
    nonNOP_action_type_table.append(action_table_row_inplan)
nonNOP_table = nonNOP_iter_table + [['' for _ in iter_table_row] ] + nonNOP_action_type_table
print(tabulate(nonNOP_table, headers=headers))