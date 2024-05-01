"""Plot the # of LLM operators vs. # iterations.
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

# Look in planning_ops.pkl. LLM-Derived Ops will have two underscores, length 3 string upon splitting on underscores.
def make_plot(domain_name, learning_name, curiosity_name, seed):
    assert 'LLM' in learning_name
    path = os.path.join(LNDR_PATH, domain_name, learning_name, curiosity_name, seed)

    iters = []
    num_llm_ops = []

    iter_dirs = []
    for iter_dir in sorted(os.listdir(path)):
        if not iter_dir.startswith('iter'): continue
        iter_dirs.append(iter_dir)
    iter_dirs = sorted(iter_dirs, key=lambda x: int(x[5:]))

    for iter_dir in iter_dirs:
        with open(os.path.join(path, iter_dir, 'planning_operators.pkl'), 'rb') as f:
            planning_ops = pickle.load(f)
        iters.append(int(iter_dir[5:]))
        n = 0
        for op in planning_ops:
            if len(op.name.split('_')) >= 3:
                n += 1
        num_llm_ops.append(n)
    plt.plot(iters, num_llm_ops)
    plt.ylabel("Number LLM Ops")
    plt.xlabel("Iteration #")
    plt.title(f'{domain_name}-{curiosity_name} Seed {seed}')
    plt.show()

if __name__ == '__main__':
    domain_name = 'Minecraft'
    learning_name = 'LLMWarmStart+LNDR'
    curiosity_name = 'GLIB_L2'
    seeds = range(910, 920)
    for seed in seeds:
        make_plot(domain_name, learning_name, curiosity_name, str(seed))