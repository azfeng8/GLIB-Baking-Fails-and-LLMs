#TODO: refactor interactive views 1, 2, and 123
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image  as mpimg
import os, pickle
import numpy as np
from pddlgym.parser import PDDLDomainParser
from collections import defaultdict

SOURCE_PATH = '/home/catalan/GLIB-Baking-Fails-and-LLMs/results/LNDR'
RESULTS_PATH = '/home/catalan/GLIB-Baking-Fails-and-LLMs/results'
SAVE_PATH = '/home/catalan/GLIB-Baking-Fails-and-LLMs/dataset_visualizations'
BABBLING_SOURCE_PATH = '/home/catalan/GLIB-Baking-Fails-and-LLMs/results/GLIB'
PDDLGYM_PATH = '/home/catalan/.virtualenvs/meng/lib/python3.10/site-packages/pddlgym/pddl'

curr_pos_view_1 = 0
curr_pos_view_2 = 0
nops_view = 0

# View 1

def view1(save_path, operators, transition_data):
    """Create and write the view1 plot from the logs to the `save_path` folder as `nops_plot.png`.
    """
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

        pie_ax.pie(ys, labels=labels, autopct='%1.3f%%')
        
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
    plt.close()

def view2(save_path, operators, transition_data, ndrs, domain_name):
    """Create a plot for each skill (view2), and save to `save_path` folder with each skill labeled by `{skill_name}.png`.

    Args:
        base_path (str): iteration folder with the operators, transition data, and NDRs.
        save_path (str): folder where the plots are saved.
    """
    # Get ground truth ops / literals
    parser = PDDLDomainParser(os.path.join(PDDLGYM_PATH, f'{domain_name.lower()}.pddl'))
    gt_op_preconds = defaultdict(set)
    for op in parser.operators.values():
        action = [l for l in op.preconds.literals if l.predicate.name in parser.actions][0]
        for lit in op.preconds.literals:
            if lit != action:
                pred_name = lit.predicate.name
                gt_op_preconds[action.predicate.name].add(pred_name)

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
        axs[0].set_title("Literals present in the initial state of non-NOPs")
        
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

        # Bold predicates in the preconditions of ground truth operators
        for label in axs[0].get_yticklabels():
            literal_str = label.get_text()
            pred_name = literal_str.split('(')[0]
            if pred_name in gt_op_preconds[action_pred.name]:
                label.set_fontweight("bold")
        
        os.makedirs(save_path, exist_ok=True)
        plt.gcf().set_size_inches(18, 14)
        plt.savefig(os.path.join(save_path, f'{action_pred.name}.png'), dpi=300)
        plt.close()


def view3(save_path, operators, transition_data, ndrs, domain_name):
    """Create a plot for each skill (view2), and save to `save_path` folder with each skill labeled by `{skill_name}.png`.

    Args:
        base_path (str): iteration folder with the operators, transition data, and NDRs.
        save_path (str): folder where the plots are saved.
    """
    # Get ground truth ops / literals
    parser = PDDLDomainParser(os.path.join(PDDLGYM_PATH, f'{domain_name.lower()}.pddl'))
    gt_op_preconds = defaultdict(set)
    for op in parser.operators.values():
        action = [l for l in op.preconds.literals if l.predicate.name in parser.actions][0]
        for lit in op.preconds.literals:
            if lit != action:
                pred_name = lit.predicate.name
                gt_op_preconds[action.predicate.name].add(pred_name)


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
        axs[0].set_title("Literals present in the initial state of NOPs")
        
        lit_counts = defaultdict(lambda: 0)
        for t in transition_data[action_pred]:
            if len(t[2]) == 0:
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

        # Bold predicates in the preconditions of ground truth operators
        for label in axs[0].get_yticklabels():
            literal_str = label.get_text()
            pred_name = literal_str.split('(')[0]
            if pred_name in gt_op_preconds[action_pred.name]:
                label.set_fontweight("bold")
               
        os.makedirs(save_path, exist_ok=True)
        plt.gcf().set_size_inches(18, 14)
        plt.savefig(os.path.join(save_path, f'{action_pred.name}-NOPs.png'), dpi=300)
        plt.close()


def interactive_view1(domain_name, curiosity_name, learning_name, seed):
    """Interactive view of the NOPs / operators plot.
    
    use right / left arrows to scroll the iteration numbers loaded.
    
    use up / down arrows to jump to the next iteration where success increases.
    
    use "i" / "o" to jump to the previous / next episode start.

    use "h" / "j" to -1 / +1 iteration.

    Args:
        domain_name (str)
        curiosity_name (str)
        seed (str)
    """
    path = os.path.join(SOURCE_PATH, domain_name, curiosity_name, seed)
    iter_dirs = []
    for dir in os.listdir(path):
        if '.txt' in dir or '.pkl' in dir: continue
        iter_dirs.append(dir)
    iter_dirs = sorted(iter_dirs, key=lambda x: int(x[5:]))


    # Initialize the plot at iter=0
    iter_dir = iter_dirs[0]
    iter_path = os.path.join(path, iter_dir)
    iter_save_path = os.path.join(SAVE_PATH, domain_name, curiosity_name, seed, iter_dir)
    filepath = os.path.join(iter_save_path, 'nops_plot.png')
    with open(os.path.join(iter_path, 'transition_data.pkl'), 'rb') as f:
        transition_data = pickle.load(f)
    if not os.path.exists(filepath):
        view1(iter_save_path, [], transition_data)

    success_increases = np.loadtxt(os.path.join(SOURCE_PATH, domain_name, curiosity_name, seed, 'success_increases.txt'))
    if len(success_increases.shape) == 1:
        success_increases = success_increases[np.newaxis, :]
    success_itrs = success_increases[:, 0].tolist()


    results_path = os.path.join(RESULTS_PATH, domain_name, learning_name, curiosity_name, f'{domain_name}_{learning_name}_{curiosity_name}_{seed}.pkl')
    with open(results_path, 'rb') as f:
        results = np.array(pickle.load(f))
        succ = results[:, 1]

    episode_start_iters = np.loadtxt(os.path.join(path, 'episode_start_iters.txt'))

    with open(os.path.join(path, 'skill_sequence.pkl'), 'rb') as f:
        skill_seq = pickle.load(f)
    
    if "GLIB" in curiosity_name:
        with open(os.path.join(BABBLING_SOURCE_PATH, domain_name, learning_name, curiosity_name, f'{seed}_babbling_stats.pkl'), 'rb') as f:
            babbling_seq = pickle.load(f)

    def key_event(e):
        global curr_pos_view_1
        nonlocal iter_dirs

        if e.key == "right":
            curr_pos_view_1 = curr_pos_view_1 + 1
        elif e.key == "left":
            curr_pos_view_1 = curr_pos_view_1 - 1
        elif e.key == 'up':
            # next success increase
            curr_pos_view_1 += 1
            curr_pos_view_1 = curr_pos_view_1 % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_view_1][5:])
            while curr_itr not in success_itrs:
                curr_pos_view_1 += 1
                curr_pos_view_1 = curr_pos_view_1 % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_view_1][5:])
        elif e.key == 'down':
            # prev success increase
            curr_pos_view_1 -= 1
            curr_pos_view_1 = curr_pos_view_1 % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_view_1][5:])
            while curr_itr not in success_itrs:
                curr_pos_view_1 -= 1
                curr_pos_view_1 = curr_pos_view_1 % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_view_1][5:])
        elif e.key == 'o':
            # next episode start
            curr_pos_view_1 += 1
            curr_pos_view_1 = curr_pos_view_1 % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_view_1][5:])
            while curr_itr not in episode_start_iters:
                curr_pos_view_1 -= 1
                curr_pos_view_1 = curr_pos_view_1 % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_view_1][5:])
        elif e.key == 'i':
             # prev episode start
            curr_pos_view_1 -= 1
            curr_pos_view_1 = curr_pos_view_1 % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_view_1][5:])
            while curr_itr not in episode_start_iters:
                curr_pos_view_1 -= 1
                curr_pos_view_1 = curr_pos_view_1 % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_view_1][5:])           
        elif e.key == 'h':
            # prev iteration
            itr = int(iter_dirs[curr_pos_view_1][5:])
            if itr > 0:
                iter_dir = f'iter_{itr - 1}'
                if iter_dir not in iter_dirs:
                    iter_dirs.append(iter_dir)
                    iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
            else:
                return
        elif e.key == 'j':
            # next iteration
            itr = int(iter_dirs[curr_pos_view_1][5:])
            if itr < int(iter_dirs[-1][5:]):
                iter_dir = f'iter_{itr + 1}'
                if iter_dir not in iter_dirs:
                    iter_dirs.append(iter_dir)
                    iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
                curr_pos_view_1 += 1
            else:
                return

        else:
            return
        curr_pos_view_1 = curr_pos_view_1 % len(iter_dirs)

        ax.cla()
        iter_dir = iter_dirs[curr_pos_view_1]
        itr_num = int(iter_dir[5:])
        iter_save_path = os.path.join(SAVE_PATH, domain_name, curiosity_name, seed, iter_dir)
        filepath = os.path.join(iter_save_path, 'nops_plot.png')
        if not os.path.exists(filepath):
            # Look ahead for the transition data, and look behind for operators and NDRs.
            curr = curr_pos_view_1
            transition_data_itr = None
            while curr < len(iter_dirs):
                if os.path.exists(os.path.join(path, iter_dirs[curr], 'transition_data.pkl')):
                    transition_data_itr = curr
                    break
                curr += 1
            if transition_data_itr is None:
                return
            curr = curr_pos_view_1
            ops_itr = None
            while curr >= 0:
                if os.path.exists(os.path.join(path, iter_dirs[curr], 'operators.pkl')) :
                    ops_itr = curr
                    break
                curr -= 1
            if ops_itr is None:
                return

            with open(os.path.join(path, iter_dirs[transition_data_itr], 'transition_data.pkl'), 'rb') as f:
                transition_data = pickle.load(f)
            with open(os.path.join(path, iter_dirs[ops_itr], 'operators.pkl'), 'rb') as f:
                ops = pickle.load(f)
            # use skill sequence to create the right dataset
            action_end = int(iter_dirs[transition_data_itr][5:])
            action_start = int(iter_dir[5:])
            for action in skill_seq[action_start + 1 : action_end + 1][::-1]:
                # LIFO
                transition_data[action.predicate].pop()
            
            view1(iter_save_path, ops, transition_data)

        img = mpimg.imread(filepath)
        ax.set_title(f"{iter_dir} : success rate {succ[int(iter_dir[5:])]}")
        if "GLIB" in curiosity_name:
            if babbling_seq[itr_num]:
                goal, plan = babbling_seq[itr_num]
                ax.set_xlabel(f'goal: {goal}\nplan: {plan}\naction: {skill_seq[itr_num]}')
        else:
            ax.set_xlabel(f'action: {skill_seq[itr_num]}')
        ax.imshow(img)
        fig.canvas.draw()

    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', key_event)
    ax = fig.add_subplot(111)
    img = mpimg.imread(filepath)
    ax.set_title(f'{iter_dirs[0]} : success rate {succ[0]}')
    plt.imshow(img)
    plt.show()

def interactive_view2(domain_name, curiosity_name, learning_name, seed):
    """Interactive plots for view2.
    
    right/left arrow keys to toggle between iterations, in order.
    
    up/down arrow keys to jump to the next iteration where success increases/decreases.

    A window for each skill is spinned up, and each time the arrow key is pressed, all of the plots are updated to that iteration.

    Args:
        domain_name (str)
        curiosity_name (str)
        seed (str)
    """

    path = os.path.join(SOURCE_PATH, domain_name, curiosity_name, seed)
    iter_dirs = []
    for dir in os.listdir(path):
        if '.txt' in dir or '.pkl' in dir: continue
        iter_dirs.append(dir)
    iter_dirs = sorted(iter_dirs, key=lambda x: int(x[5:]))

    #spin up a figure for each skill eventually seen
    idx = len(iter_dirs) - 1
    while not os.path.exists(os.path.join(path, iter_dirs[idx], 'transition_data.pkl')):
        idx -= 1
    with open(os.path.join(path, iter_dirs[idx], 'transition_data.pkl'),'rb')  as f:
        transition_data = pickle.load(f)
    actions = []
    figs = {}
    for action_pred in transition_data:
        actions.append(action_pred.name)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        figs[action_pred.name] = (fig, ax)

    # Create `succ`, an array of success rates for each iter_dir logged
    success_increases = np.loadtxt(os.path.join(SOURCE_PATH, domain_name, curiosity_name, seed, 'success_increases.txt'))
    success_itrs = success_increases[:, 0].tolist()

    results_path = os.path.join(RESULTS_PATH, domain_name, learning_name, curiosity_name, f'{domain_name}_{learning_name}_{curiosity_name}_{seed}.pkl')
    with open(results_path, 'rb') as f:
        results = np.array(pickle.load(f))
        succ = results[:, 1]

    episode_start_iters = np.loadtxt(os.path.join(path, 'episode_start_iters.txt'))

    with open(os.path.join(path, 'skill_sequence.pkl'), 'rb') as f:
        skill_seq = pickle.load(f)
    
    if "GLIB" in curiosity_name:
        with open(os.path.join(BABBLING_SOURCE_PATH, domain_name, learning_name, curiosity_name, f'{seed}_babbling_stats.pkl'), 'rb') as f:
            babbling_seq = pickle.load(f)


    def key_event(e):
        global curr_pos_view_2
        nonlocal iter_dirs

        if e.key == "right":
            curr_pos_view_2 = curr_pos_view_2 + 1
        elif e.key == "left":
            curr_pos_view_2 = curr_pos_view_2 - 1
        elif e.key == 'up':
            curr_pos_view_2 += 1
            curr_pos_view_2 = curr_pos_view_2 % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_view_2][5:])
            while curr_itr not in success_itrs:
                curr_pos_view_2 += 1
                curr_pos_view_2 = curr_pos_view_2 % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_view_2][5:])
        elif e.key == 'down':
            curr_pos_view_2 -= 1
            curr_pos_view_2 = curr_pos_view_2 % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_view_2][5:])
            while curr_itr not in success_itrs:
                curr_pos_view_2 -= 1
                curr_pos_view_2 = curr_pos_view_2 % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_view_2][5:])
        elif e.key == 'o':
            # next episode start
            curr_pos_view_2 += 1
            curr_pos_view_2 = curr_pos_view_2 % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_view_2][5:])
            while curr_itr not in episode_start_iters:
                curr_pos_view_2 -= 1
                curr_pos_view_2 = curr_pos_view_2 % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_view_2][5:])
        elif e.key == 'i':
             # prev episode start
            curr_pos_view_2 -= 1
            curr_pos_view_2 = curr_pos_view_2 % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_view_2][5:])
            while curr_itr not in episode_start_iters:
                curr_pos_view_2 -= 1
                curr_pos_view_2 = curr_pos_view_2 % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_view_2][5:])           
        elif e.key == 'h':
            # prev iteration
            itr = int(iter_dirs[curr_pos_view_2][5:])
            if itr > 0:
                iter_dir = f'iter_{itr - 1}'
                if iter_dir not in iter_dirs:
                    iter_dirs.append(iter_dir)
                    iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
            else:
                return
        elif e.key == 'j':
            # next iteration
            itr = int(iter_dirs[curr_pos_view_2][5:])
            if itr < int(iter_dirs[-1][5:]):
                iter_dir = f'iter_{itr + 1}'
                if iter_dir not in iter_dirs:
                    iter_dirs.append(iter_dir)
                    iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
                curr_pos_view_2 += 1
            else:
                return
        else:
            return

        curr_pos_view_2 = curr_pos_view_2 % len(iter_dirs)
        iter_dir = iter_dirs[curr_pos_view_2]
        iter_num = int(iter_dir[5:])
        iter_save_path = os.path.join(SAVE_PATH, domain_name, curiosity_name, seed, iter_dir)
 
        curr = curr_pos_view_2
        transition_data_itr = None
        while curr < len(iter_dirs):
            if os.path.exists(os.path.join(path, iter_dirs[curr], 'transition_data.pkl')):
                transition_data_itr = curr
                break
            curr += 1
        if transition_data_itr is None:
            return
        curr = curr_pos_view_2
        ops_itr = None
        while curr >= 0:
            if os.path.exists(os.path.join(path, iter_dirs[curr], 'operators.pkl')) :
                ops_itr = curr
                break
            curr -= 1
        if ops_itr is None:
            return

        with open(os.path.join(path, iter_dirs[transition_data_itr], 'transition_data.pkl'), 'rb') as f:
            transition_data = pickle.load(f)

        # use skill sequence to create the right dataset
        action_end = int(iter_dirs[transition_data_itr][5:])
        action_start = iter_num
        for action in skill_seq[action_start + 1 : action_end + 1][::-1]:
            # LIFO
            transition_data[action.predicate].pop()
            if len(transition_data[action.predicate]) == 0:
                del transition_data[action.predicate]
 
        actions_to_plot = []
        create = False
        for action in transition_data:
            filepath = os.path.join(iter_save_path, f'{action.name}.png')
            if not os.path.exists(filepath):
                create = True
            actions_to_plot.append(action.name)

        if create:
            with open(os.path.join(path, iter_dirs[ops_itr], 'operators.pkl'), 'rb') as f:
                ops = pickle.load(f)
            with open(os.path.join(path, iter_dirs[ops_itr], 'ndrs.pkl'), 'rb') as f:
                ndrs = pickle.load(f)
            view2(iter_save_path, ops, transition_data, ndrs, domain_name)

        for act, figax in figs.items():
            fig, ax = figax
            ax.cla()
            if act in actions_to_plot:
                filepath = os.path.join(iter_save_path, f'{act}.png')
                print(filepath)
                img = mpimg.imread(filepath)
                ax.set_title(f"{iter_dir} : success rate {succ[iter_num]}")
                if "GLIB" in curiosity_name:
                    if babbling_seq[iter_num]:
                        goal, plan = babbling_seq[iter_num]
                        ax.set_xlabel(f'goal: {goal}\nplan: {plan}\naction: {skill_seq[iter_num]}')
                else:
                    ax.set_xlabel(f'action: {skill_seq[iter_num]}')
        
                ax.imshow(img)
                fig.canvas.draw()

    # Initialize the plots for each skill
    iter_dir = iter_dirs[0]
    iter_save_path = os.path.join(SAVE_PATH, domain_name, curiosity_name, seed, iter_dir)

    with open(os.path.join(path, iter_dirs[0], 'transition_data.pkl'),'rb')  as f:
        transition_data = pickle.load(f)
    with open(os.path.join(path, iter_dirs[0], 'operators.pkl'),'rb')  as f:
        ops = pickle.load(f)
    with open(os.path.join(path, iter_dirs[0], 'ndrs.pkl'),'rb')  as f:
        ndrs = pickle.load(f)

    create = False
    init_actions = []
    for act in transition_data:
        filepath = os.path.join(iter_save_path, f'{act.name}.png')
        init_actions.append(act.name)
        if not os.path.exists(filepath):
            create = True
    if create:
        view2(iter_save_path, ops, transition_data, ndrs, domain_name)

    for act, figax in figs.items():
        fig, ax = figax
        fig.canvas.mpl_connect('key_press_event', key_event)
        if act in init_actions:
            filepath = os.path.join(iter_save_path, f'{act}.png')
            img = mpimg.imread(filepath)
            ax.set_title(f'{iter_dirs[0]} : success rate {succ[0]}')
            ax.imshow(img)
    plt.show()

def interactive_view_123(domain_name, curiosity_name, learning_name, seed):
    """Interactive View 1, 2, and 3.

    use right/left arrow keys to toggle between iterations.
    use up/down arrow keys to toggle between success increase/decrease iterations.
    
    The view1 plot is independent of the view2 plots, which all change from one keystroke.
    """

    path = os.path.join(SOURCE_PATH, domain_name, curiosity_name, seed)
    iter_dirs = []
    for dir in os.listdir(path):
        if '.txt' in dir or '.pkl' in dir: continue
        iter_dirs.append(dir)
    iter_dirs = sorted(iter_dirs, key=lambda x: int(x[5:]))

    #spin up a figure for each skill eventually seen
    idx = len(iter_dirs) - 1
    while not os.path.exists(os.path.join(path, iter_dirs[idx], 'transition_data.pkl')):
        idx -= 1
    with open(os.path.join(path, iter_dirs[idx], 'transition_data.pkl'),'rb')  as f:
        transition_data = pickle.load(f)
    actions = []
    figs = {}
    for action_pred in transition_data:
        actions.append(action_pred.name)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        figs[action_pred.name] = (fig, ax)


    # Initialize the plot at iter=0
    iter_dir = iter_dirs[0]
    iter_path = os.path.join(path, iter_dir)
    iter_save_path = os.path.join(SAVE_PATH, domain_name, curiosity_name, seed, iter_dir)
    filepath = os.path.join(iter_save_path, 'nops_plot.png')
    with open(os.path.join(iter_path, 'transition_data.pkl'), 'rb') as f:
        transition_data = pickle.load(f)
    if not os.path.exists(filepath):
        view1(iter_save_path, [], transition_data)

    success_increases = np.loadtxt(os.path.join(SOURCE_PATH, domain_name, curiosity_name, seed, 'success_increases.txt'))
    if len(success_increases.shape) == 1:
        success_increases = success_increases[np.newaxis, :]
    success_itrs = success_increases[:, 0].tolist()


    results_path = os.path.join(RESULTS_PATH, domain_name, learning_name, curiosity_name, f'{domain_name}_{learning_name}_{curiosity_name}_{seed}.pkl')
    with open(results_path, 'rb') as f:
        results = np.array(pickle.load(f))
        succ = results[:, 1]

    episode_start_iters = np.loadtxt(os.path.join(path, 'episode_start_iters.txt'))

    with open(os.path.join(path, 'skill_sequence.pkl'), 'rb') as f:
        skill_seq = pickle.load(f)
    
    if "GLIB" in curiosity_name:
        with open(os.path.join(BABBLING_SOURCE_PATH, domain_name, learning_name, curiosity_name, f'{seed}_babbling_stats.pkl'), 'rb') as f:
            babbling_seq = pickle.load(f)


    def key_event_view_2(e):
        global curr_pos_view_2
        global nops_view
        nonlocal iter_dirs

        if e.key == "right":
            curr_pos_view_2 = curr_pos_view_2 + 1
        elif e.key == "left":
            curr_pos_view_2 = curr_pos_view_2 - 1
        elif e.key == 'up':
            curr_pos_view_2 += 1
            curr_pos_view_2 = curr_pos_view_2 % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_view_2][5:])
            while curr_itr not in success_itrs:
                curr_pos_view_2 += 1
                curr_pos_view_2 = curr_pos_view_2 % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_view_2][5:])
        elif e.key == 'down':
            curr_pos_view_2 -= 1
            curr_pos_view_2 = curr_pos_view_2 % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_view_2][5:])
            while curr_itr not in success_itrs:
                curr_pos_view_2 -= 1
                curr_pos_view_2 = curr_pos_view_2 % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_view_2][5:])
        elif e.key == 'r':
            nops_view = 1 - nops_view
        elif e.key == 'o':
            # next episode start
            curr_pos_view_2 += 1
            curr_pos_view_2 = curr_pos_view_2 % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_view_2][5:])
            while curr_itr not in episode_start_iters:
                curr_pos_view_2 -= 1
                curr_pos_view_2 = curr_pos_view_2 % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_view_2][5:])
        elif e.key == 'i':
             # prev episode start
            curr_pos_view_2 -= 1
            curr_pos_view_2 = curr_pos_view_2 % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_view_2][5:])
            while curr_itr not in episode_start_iters:
                curr_pos_view_2 -= 1
                curr_pos_view_2 = curr_pos_view_2 % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_view_2][5:])           
        elif e.key == 'h':
            # prev iteration
            itr = int(iter_dirs[curr_pos_view_2][5:])
            if itr > 0:
                iter_dir = f'iter_{itr - 1}'
                if iter_dir not in iter_dirs:
                    iter_dirs.append(iter_dir)
                    iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
            else:
                return
        elif e.key == 'j':
            # next iteration
            itr = int(iter_dirs[curr_pos_view_2][5:])
            if itr < int(iter_dirs[-1][5:]):
                iter_dir = f'iter_{itr + 1}'
                if iter_dir not in iter_dirs:
                    iter_dirs.append(iter_dir)
                    iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
                curr_pos_view_2 += 1
            else:
                return
        else:
            return

        curr_pos_view_2 = curr_pos_view_2 % len(iter_dirs)
        iter_dir = iter_dirs[curr_pos_view_2]
        iter_num = int(iter_dir[5:])
        iter_save_path = os.path.join(SAVE_PATH, domain_name, curiosity_name, seed, iter_dir)
 
        curr = curr_pos_view_2
        transition_data_itr = None
        while curr < len(iter_dirs):
            if os.path.exists(os.path.join(path, iter_dirs[curr], 'transition_data.pkl')):
                transition_data_itr = curr
                break
            curr += 1
        if transition_data_itr is None:
            return
        curr = curr_pos_view_2
        ops_itr = None
        while curr >= 0:
            if os.path.exists(os.path.join(path, iter_dirs[curr], 'operators.pkl')) :
                ops_itr = curr
                break
            curr -= 1
        if ops_itr is None:
            return

        with open(os.path.join(path, iter_dirs[transition_data_itr], 'transition_data.pkl'), 'rb') as f:
            transition_data = pickle.load(f)

        # use skill sequence to create the right dataset
        action_end = int(iter_dirs[transition_data_itr][5:])
        action_start = iter_num
        for action in skill_seq[action_start + 1 : action_end + 1][::-1]:
            # LIFO
            transition_data[action.predicate].pop()
            if len(transition_data[action.predicate]) == 0:
                del transition_data[action.predicate]
 
        actions_to_plot = []
        create_view2 = False
        create_view3 = False
        for action in transition_data:
            filepath = os.path.join(iter_save_path, f'{action.name}.png')
            view3_filepath = os.path.join(iter_save_path, f'{action.name}-NOPs.png')
            if not os.path.exists(filepath):
                create_view2 = True
            if not os.path.exists(view3_filepath):
                create_view3 = True
            actions_to_plot.append(action.name)

        if create_view3 or create_view2:
            with open(os.path.join(path, iter_dirs[ops_itr], 'operators.pkl'), 'rb') as f:
                ops = pickle.load(f)
            with open(os.path.join(path, iter_dirs[ops_itr], 'ndrs.pkl'), 'rb') as f:
                ndrs = pickle.load(f)
        if create_view2:
            view2(iter_save_path, ops, transition_data, ndrs, domain_name)
        if create_view3:
            view3(iter_save_path, ops, transition_data, ndrs, domain_name)

        for act, figax in figs.items():
            fig, ax = figax
            ax.cla()
            if act in actions_to_plot:
                if nops_view:
                    filepath = os.path.join(iter_save_path, f'{act}-NOPs.png')
                else:
                    filepath = os.path.join(iter_save_path, f'{act}.png')
                print(filepath)
                img = mpimg.imread(filepath)
                ax.set_title(f"{iter_dir} : success rate {succ[iter_num]}")
                if "GLIB" in curiosity_name:
                    if babbling_seq[iter_num]:
                        goal, plan = babbling_seq[iter_num]
                        ax.set_xlabel(f'goal: {goal}\nplan: {plan}\naction: {skill_seq[iter_num]}')
                else:
                    ax.set_xlabel(f'action: {skill_seq[iter_num]}')
        
                ax.imshow(img)
                fig.canvas.draw()

    # Initialize the plots for each skill
    iter_dir = iter_dirs[0]
    iter_save_path = os.path.join(SAVE_PATH, domain_name, curiosity_name, seed, iter_dir)

    with open(os.path.join(path, iter_dirs[0], 'transition_data.pkl'),'rb')  as f:
        transition_data = pickle.load(f)
    with open(os.path.join(path, iter_dirs[0], 'operators.pkl'),'rb')  as f:
        ops = pickle.load(f)
    with open(os.path.join(path, iter_dirs[0], 'ndrs.pkl'),'rb')  as f:
        ndrs = pickle.load(f)

    create_view2 = False
    create_view3 = False
    init_actions = []
    for act in transition_data:
        filepath = os.path.join(iter_save_path, f'{act.name}.png')
        view3_filepath = os.path.join(iter_save_path, f'{act.name}-NOPs.png')
        init_actions.append(act.name)
        if not os.path.exists(filepath):
            create_view2 = True
        if not os.path.exists(view3_filepath):
            create_view3 = True
    if create_view2:
        view2(iter_save_path, ops, transition_data, ndrs, domain_name)
    if create_view3:
        view3(iter_save_path, ops, transition_data, ndrs, domain_name)

    for act, figax in figs.items():
        fig, ax = figax
        fig.canvas.mpl_connect('key_press_event', key_event_view_2)
        if act in init_actions:
            filepath = os.path.join(iter_save_path, f'{act}.png')
            img = mpimg.imread(filepath)
            ax.set_title(f'{iter_dirs[0]} : success rate {succ[0]}')
            ax.imshow(img)

    def key_event_view_1(e):
        global curr_pos_view_1
        nonlocal iter_dirs

        if e.key == "right":
            curr_pos_view_1 = curr_pos_view_1 + 1
        elif e.key == "left":
            curr_pos_view_1 = curr_pos_view_1 - 1
        elif e.key == 'up':
            # next success increase
            curr_pos_view_1 += 1
            curr_pos_view_1 = curr_pos_view_1 % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_view_1][5:])
            while curr_itr not in success_itrs:
                curr_pos_view_1 += 1
                curr_pos_view_1 = curr_pos_view_1 % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_view_1][5:])
        elif e.key == 'down':
            # prev success increase
            curr_pos_view_1 -= 1
            curr_pos_view_1 = curr_pos_view_1 % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_view_1][5:])
            while curr_itr not in success_itrs:
                curr_pos_view_1 -= 1
                curr_pos_view_1 = curr_pos_view_1 % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_view_1][5:])
        elif e.key == 'o':
            # next episode start
            curr_pos_view_1 += 1
            curr_pos_view_1 = curr_pos_view_1 % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_view_1][5:])
            while curr_itr not in episode_start_iters:
                curr_pos_view_1 -= 1
                curr_pos_view_1 = curr_pos_view_1 % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_view_1][5:])
        elif e.key == 'i':
             # prev episode start
            curr_pos_view_1 -= 1
            curr_pos_view_1 = curr_pos_view_1 % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_view_1][5:])
            while curr_itr not in episode_start_iters:
                curr_pos_view_1 -= 1
                curr_pos_view_1 = curr_pos_view_1 % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_view_1][5:])           
        elif e.key == 'h':
            # prev iteration
            itr = int(iter_dirs[curr_pos_view_1][5:])
            if itr > 0:
                iter_dir = f'iter_{itr - 1}'
                if iter_dir not in iter_dirs:
                    iter_dirs.append(iter_dir)
                    iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
            else:
                return
        elif e.key == 'j':
            # next iteration
            itr = int(iter_dirs[curr_pos_view_1][5:])
            if itr < int(iter_dirs[-1][5:]):
                iter_dir = f'iter_{itr + 1}'
                if iter_dir not in iter_dirs:
                    iter_dirs.append(iter_dir)
                    iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
                curr_pos_view_1 += 1
            else:
                return

        else:
            return
        curr_pos_view_1 = curr_pos_view_1 % len(iter_dirs)

        ax.cla()
        iter_dir = iter_dirs[curr_pos_view_1]
        itr_num = int(iter_dir[5:])
        iter_save_path = os.path.join(SAVE_PATH, domain_name, curiosity_name, seed, iter_dir)
        filepath = os.path.join(iter_save_path, 'nops_plot.png')
        if not os.path.exists(filepath):
            # Look ahead for the transition data, and look behind for operators and NDRs.
            curr = curr_pos_view_1
            transition_data_itr = None
            while curr < len(iter_dirs):
                if os.path.exists(os.path.join(path, iter_dirs[curr], 'transition_data.pkl')):
                    transition_data_itr = curr
                    break
                curr += 1
            if transition_data_itr is None:
                return
            curr = curr_pos_view_1
            ops_itr = None
            while curr >= 0:
                if os.path.exists(os.path.join(path, iter_dirs[curr], 'operators.pkl')) :
                    ops_itr = curr
                    break
                curr -= 1
            if ops_itr is None:
                return

            with open(os.path.join(path, iter_dirs[transition_data_itr], 'transition_data.pkl'), 'rb') as f:
                transition_data = pickle.load(f)
            with open(os.path.join(path, iter_dirs[ops_itr], 'operators.pkl'), 'rb') as f:
                ops = pickle.load(f)
            # use skill sequence to create the right dataset
            action_end = int(iter_dirs[transition_data_itr][5:])
            action_start = int(iter_dir[5:])
            for action in skill_seq[action_start + 1 : action_end + 1][::-1]:
                # LIFO
                transition_data[action.predicate].pop()
            
            view1(iter_save_path, ops, transition_data)

        img = mpimg.imread(filepath)
        ax.set_title(f"{iter_dir} : success rate {succ[int(iter_dir[5:])]}")
        if "GLIB" in curiosity_name:
            if babbling_seq[itr_num]:
                goal, plan = babbling_seq[itr_num]
                ax.set_xlabel(f'goal: {goal}\nplan: {plan}\naction: {skill_seq[itr_num]}')
        else:
            ax.set_xlabel(f'action: {skill_seq[itr_num]}')
        ax.imshow(img)
        fig.canvas.draw()


    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', key_event_view_1)
    ax = fig.add_subplot(111)
    img = mpimg.imread(filepath)
    ax.set_title(f'{iter_dirs[0]} : success rate {succ[0]}')
    plt.imshow(img)
    plt.show()



if __name__ == "__main__":
        
    domain_name = 'Blocks'
    # curiosity_name = 'random'
    curiosity_name = 'GLIB_L2'
    learning_name = 'LNDR'
    # seeds = [str(s) for s in range(400, 405)]
    seed = '405'
    # interactive_view_123(domain_name, curiosity_name, seed)
    # interactive_view2(domain_name, curiosity_name, seed)
    interactive_view_123(domain_name, curiosity_name, learning_name, seed)

    # with open(os.path.join(BABBLING_SOURCE_PATH, domain_name, learning_name, curiosity_name, f'{seed}_babbling_stats.pkl'), 'rb') as f:
        # stats = pickle.load(f)
    # print(stats[80:90])