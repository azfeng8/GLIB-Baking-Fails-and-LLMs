import matplotlib.pyplot as plt
import os, pickle
import numpy as np
from collections import defaultdict

# View 1

def view1(base_path, save_path):
    with open(os.path.join(base_path, 'operators.pkl'), 'rb') as f:
        operators = pickle.load(f)

    with open(os.path.join(base_path, 'transition_data.pkl'), 'rb') as f:
        transition_data = pickle.load(f)

    ncols = len(transition_data)
    col_names = [t.name for t in transition_data]
    row_names = ["NOPs ratio", "", "Operators"]
    fig, axs = plt.subplots(ncols=ncols, nrows=3)

    for ax, col in zip(axs[0], col_names):
        ax.set_title(col)
    for ax, row in zip(axs[:, 0], row_names):
        ax.set_ylabel(row, size='large')


    for i,action_pred in enumerate(transition_data):
        nops = 0
        total = len(transition_data[action_pred])
        for t in transition_data[action_pred]:
            if len(t[2]) == 0:
                nops += 1
        ys = [nops, total - nops]
        y_pos = np.arange(len(ys))
        labels = ["NOPs", "non-NOPs"]
        axs[0,i].barh(y_pos, ys)
        axs[0,i].set_yticks(y_pos, labels=labels)
        axs[0,i].set_xlabel("Frequency")
        axs[0,i].invert_yaxis()  # labels read top-to-bottom

        axs[1, i].pie(ys, labels=labels, autopct='%1.1f%%')
        
        ops = []
        for o in operators:
            for lit in  o.preconds.literals:
                if lit.predicate.name == action_pred.name:
                    ops.append(o.pddl_str())
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axs[2,i].text(0,1, '\n'.join(ops),  fontsize=8, verticalalignment='top', bbox=props, wrap=True)

    plt.gcf().set_size_inches(18, 14)
    plt.savefig(os.path.join(save_path, 'nops_plot.png'))


def view2(base_path, save_path):
    with open(os.path.join(base_path, 'operators.pkl'), 'rb') as f:
        operators = pickle.load(f)

    with open(os.path.join(base_path, 'transition_data.pkl'), 'rb') as f:
        transition_data = pickle.load(f)

    with open(os.path.join(base_path, 'ndrs.pkl'), 'rb') as f:
        ndrs = pickle.load(f)
    
    #TODO: alternate color of different literals
    colors = '#1f77b4', '#ff7f0e'
    for action_pred in transition_data:
        fig, axs = plt.subplots(nrows=1, ncols=3)
         
        ops = []
        for o in operators:
            for lit in  o.preconds.literals:
                if lit.predicate.name == action_pred.name:
                    ops.append(o.pddl_str())
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axs[1].text(0,1, '\n\n'.join(ops),  fontsize=11, verticalalignment='top', bbox=props, wrap=True)
        axs[2].text(0, 1, '\n\n'.join([str(rule) for rule in ndrs[action_pred]]), fontsize=11, verticalalignment='top', bbox=props, wrap=True)
        fig.suptitle(action_pred)
        axs[1].set_title('Operators')
        axs[2].set_title('NDRs')
        axs[0].set_title("Literals present in the precondition of non-NOPs")
        
        lit_counts = defaultdict(lambda: 0)
        for t in transition_data[action_pred]:
            if len(t[2]) == 0:
                continue
            for lit in t[0]:
                lit_counts[lit] += 1
        ys = []
        labels = []
        for l in sorted(lit_counts):
            ys.append(lit_counts[l])
            labels.append(l)
        y_pos = np.arange(len(ys))
        axs[0].barh(y_pos, ys)
        axs[0].set_yticks(y_pos, labels=labels)
        axs[0].set_xlabel("Frequency")
        axs[0].invert_yaxis()
        
        plt.gcf().set_size_inches(18, 14)
        plt.savefig(os.path.join(save_path, f'{action_pred.name}.png'), dpi=300)

root_source_path = '/home/catalan/GLIB-Baking-Fails-and-LLMs/results/LNDR'

domain_name = 'Blocks'
curiosity = 'random'
seed = 400

root_save_path = '/home/catalan/GLIB-Baking-Fails-and-LLMs/dataset_visualizations'
for iter_num in [101, 35, 45, 286]:
    save_path = os.path.join(root_save_path, domain_name, curiosity, str(seed), f'iter_{iter_num}')
    os.makedirs(save_path, exist_ok=True)
    source_path = os.path.join(root_source_path, domain_name, curiosity, str(seed), f'iter_{iter_num}')
    view1(source_path, save_path)
    view2(source_path, save_path)