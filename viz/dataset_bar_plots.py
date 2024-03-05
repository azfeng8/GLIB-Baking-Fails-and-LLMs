import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image  as mpimg
import os, pickle
import numpy as np
from collections import defaultdict

SOURCE_PATH = '/home/catalan/GLIB-Baking-Fails-and-LLMs/results/LNDR'
SAVE_PATH = '/home/catalan/GLIB-Baking-Fails-and-LLMs/dataset_visualizations'


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

    if ncols > 1:
        for ax, col in zip(axs[0], col_names):
            ax.set_title(col)

        for ax, row in zip(axs[:, 0], row_names):
            ax.set_ylabel(row, size='large')
    else:
        axs[0].set_title(col_names[0]) 
        for ax, row in zip(axs, row_names):
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
        if ncols > 1:
            bar_ax = axs[0, i]
            pie_ax = axs[1, i]
            ops_ax = axs[2, i]
        else:
            bar_ax = axs[0]
            pie_ax = axs[1]
            ops_ax = axs[2]

        bar_ax.barh(y_pos, ys)
        bar_ax.set_yticks(y_pos, labels=labels)
        bar_ax.set_xlabel("Frequency")
        bar_ax.invert_yaxis()  # labels read top-to-bottom

        pie_ax.pie(ys, labels=labels, autopct='%1.1f%%')
        
        ops = []
        for o in operators:
            for lit in  o.preconds.literals:
                if lit.predicate.name == action_pred.name:
                    ops.append(o.pddl_str())
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ops_ax.text(0,1, '\n'.join(ops),  fontsize=8, verticalalignment='top', bbox=props, wrap=True)

    plt.gcf().set_size_inches(18, 14)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'nops_plot.png'), dpi=300)

def view2(base_path, save_path):
    """Create a plot for each skill.

    Args:
        base_path (str): iteration folder with the operators, transition data, and NDRs.
        save_path (str): folder where the plots are saved.
    """

    with open(os.path.join(base_path, 'operators.pkl'), 'rb') as f:
        operators = pickle.load(f)

    with open(os.path.join(base_path, 'transition_data.pkl'), 'rb') as f:
        transition_data = pickle.load(f)

    with open(os.path.join(base_path, 'ndrs.pkl'), 'rb') as f:
        ndrs = pickle.load(f)
    
    # alternate color of different predicates to see better
    colors = '#1f77b4', '#ff7f0e'
    for action_pred in transition_data:
        fig, axs = plt.subplots(nrows=1, ncols=3)
         
        ops = []
        for o in operators:
            for lit in o.preconds.literals:
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
        plot_color = []
        color_i = 0
        last_predicate = None
        for l in sorted(lit_counts):
            ys.append(lit_counts[l])
            labels.append(l)
            if last_predicate is None or last_predicate != l.predicate.name:
                color_i = 1 - color_i
            plot_color.append(colors[color_i])
            last_predicate = l.predicate.name
        y_pos = np.arange(len(ys))
        axs[0].barh(y_pos, ys, color=plot_color)
        axs[0].set_yticks(y_pos, labels=labels)
        axs[0].set_xlabel("Frequency")
        axs[0].invert_yaxis()
        
        os.makedirs(save_path, exist_ok=True)
        plt.gcf().set_size_inches(18, 14)
        plt.savefig(os.path.join(save_path, f'{action_pred.name}.png'), dpi=1300)


def interactive_view1(domain_name, curiosity_name, seed):
    """Interactive view of the NOPs / operators plot.
    
    use right / left arrows to toggle the iteration number.
    
    use up / down arrows to jump to the next iteration where success increases.

    Args:
        domain_name (str)
        curiosity_name (str)
        seed (str)
    """
    path = os.path.join(SOURCE_PATH, domain_name, curiosity_name)
    seed_path = os.path.join(path, seed)
    iter_dirs = os.listdir(seed_path)
    iter_dirs.remove('success_increases.txt')
    iter_dirs = sorted(iter_dirs, key=lambda x: int(x[5:]))


    # Initialize the plot at iter=0
    iter_dir = iter_dirs[0]
    iter_path = os.path.join(seed_path, iter_dir)
    iter_save_path = os.path.join(SAVE_PATH, domain_name, curiosity_name, seed, iter_dir)
    filepath = os.path.join(iter_save_path, 'nops_plot.png')
    if not os.path.exists(filepath):
        view1(iter_path, iter_save_path)

    success_increases = np.loadtxt(os.path.join(SOURCE_PATH, domain_name, curiosity_name, seed, 'success_increases.txt'))
    success_itrs, successes = zip(success_increases)
    success_itrs, successes = list(success_itrs), list(successes)
    successes.insert(0,0)
    succ = []
    i = 0 
    for j, iterdir in enumerate(iter_dirs):
        if iterdir == 'success_increases.txt': continue
        if int(iterdir[5:]) in success_itrs:
            i+=1
        succ.append(successes[i])

    curr_pos = 0
    def key_event(e):
        global curr_pos

        if e.key == "right":
            curr_pos = curr_pos + 1
        elif e.key == "left":
            curr_pos = curr_pos - 1
        elif e.key == 'up':
            curr_itr = int(iter_dirs[curr_pos][5:])
            while curr_itr not in success_itrs:
                curr_pos += 1
                curr_pos = curr_pos % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos][5:])
        elif e.key == 'down':
            curr_itr = int(iter_dirs[curr_pos][5:])
            while curr_itr not in success_itrs:
                curr_pos -= 1
                curr_pos = curr_pos % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos][5:])
        else:
            return
        curr_pos = curr_pos % len(iter_dirs)

        ax.cla()
        iter_dir = iter_dirs[curr_pos]
        iter_path = os.path.join(seed_path, iter_dir)
        iter_save_path = os.path.join(SAVE_PATH, domain_name, curiosity_name, seed, iter_dir)
        filepath = os.path.join(iter_save_path, 'nops_plot.png')
        if not os.path.exists(filepath):
            view1(iter_path, iter_save_path)
        img = mpimg.imread(filepath)
        ax.set_title(f"{iter_dir} : success rate {succ[curr_pos]}")
        ax.imshow(img)
        fig.canvas.draw()

    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', key_event)
    ax = fig.add_subplot(111)
    img = mpimg.imread(filepath)
    ax.set_title(f'{iter_dirs[0]} : success rate {succ[0]}')
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
        
    domain_name = 'Blocks'
    curiosity_name = 'GLIB_L2'
    seed = '400'
# root_source_path = '/home/catalan/GLIB-Baking-Fails-and-LLMs/results/LNDR'

# domain_name = 'Blocks'
# curiosity = 'GLIB_L2'
# seed = 402

# root_save_path = '/home/catalan/GLIB-Baking-Fails-and-LLMs/dataset_visualizations'
# # for iter_num in [101, 35, 45, 286]:
# for iter_num in [53, 80, 82, 338]:
#     save_path = os.path.join(root_save_path, domain_name, curiosity, str(seed), f'iter_{iter_num}')
#     os.makedirs(save_path, exist_ok=True)
#     source_path = os.path.join(root_source_path, domain_name, curiosity, str(seed), f'iter_{iter_num}')
#     view1(source_path, save_path)
#     view2(source_path, save_path)