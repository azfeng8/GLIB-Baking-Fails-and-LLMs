import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image  as mpimg
import os, pickle
import numpy as np
from pddlgym.parser import PDDLDomainParser
import gym, pddlgym
from collections import defaultdict

# How many operators to show on view1 before run out of visual space for displaying them, and instead write the operators to separate files.
OPERATOR_SPACE_LIMIT = 4

# SOURCE_PATH = '/home/catalan/GLIB-Baking-Fails-and-LLMs/results_openstack/results/LNDR'
# RESULTS_PATH = '/home/catalan/GLIB-Baking-Fails-and-LLMs/results_openstack/results'
# BABBLING_SOURCE_PATH = '/home/catalan/GLIB-Baking-Fails-and-LLMs/results_openstack/results/GLIB'

SOURCE_PATH = '/home/catalan/GLIB-Baking-Fails-and-LLMs/results/LNDR'
RESULTS_PATH = '/home/catalan/GLIB-Baking-Fails-and-LLMs/results'
BABBLING_SOURCE_PATH = '/home/catalan/GLIB-Baking-Fails-and-LLMs/results/GLIB'

SAVE_PATH = '/home/catalan/GLIB-Baking-Fails-and-LLMs/dataset_visualizations'
PDDLGYM_PATH = '/home/catalan/.virtualenvs/meng/lib/python3.10/site-packages/pddlgym/pddl'

curr_pos_view_1 = 0
curr_pos_view_2 = 0
curr_pos_views = 0
nops_view = 0
planning_ops_view = 0

def view1(save_path, operators, transition_data, domain_name, is_learned_ops):
    """Create and write the view1 plot from the logs to the `save_path` folder as `nops_plot.png`.
    """
    env = pddlgym.make(f"PDDLEnv{domain_name}-v0")
    cols = set()
    actions = set()
    for o in operators:
        for lit in o.preconds.literals:
            if lit.predicate in env.action_space.predicates:
                cols.add(lit.predicate)
                actions.add(lit.predicate)
        
    for act_pred in transition_data:
        cols.add(act_pred)
    cols = sorted(cols)
    ncols = len(cols)
    col_names = [t.name for t in cols]
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

    for i, action_pred in enumerate(cols):

        if ncols > 1:
            bar_ax = axs[0, i]
            pie_ax = axs[1, i]
            ops_ax = axs[2, i]
        else:
            bar_ax = axs[0]
            pie_ax = axs[1]
            ops_ax = axs[2]

        if action_pred in transition_data:
            nops = 0
            total = len(transition_data[action_pred])
            for t in transition_data[action_pred]:
                if len(t[2]) == 0:
                    nops += 1
            ys = [nops, total - nops]
            y_pos = np.arange(len(ys))
            labels = ["NOPs", "non-NOPs"]

            bar_ax.barh(y_pos, ys)
            bar_ax.set_yticks(y_pos, labels=labels)
            bar_ax.set_xlabel("Frequency")
            bar_ax.invert_yaxis()  # labels read top-to-bottom

            if total != 0:
                pie_ax.pie(ys, labels=labels, autopct='%1.3f%%')
        
        if action_pred in actions:
            ops = []
            for o in operators:
                for lit in o.preconds.literals:
                    if lit.predicate.name == action_pred.name:
                        ops.append(o.pddl_str() + '\n')
            
            if len(ops) < OPERATOR_SPACE_LIMIT:
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ops_ax.text(0,1, ''.join(ops),  fontsize=8, verticalalignment='top', bbox=props, wrap=True)
            else:
                fname = os.path.join(save_path, f'ops_{action_pred.name}.txt')
                with open(fname, 'w') as f:
                    f.writelines(ops)
                print(f"Saved ops for action {action_pred.name} to: ", fname)

    plt.gcf().set_size_inches(28, 14)
    os.makedirs(save_path, exist_ok=True)
    if is_learned_ops:
        plt.savefig(os.path.join(save_path, 'nops_plot_learned_ops.png'), dpi=300)
    else:
        plt.savefig(os.path.join(save_path, 'nops_plot_planning_ops.png'), dpi=300)
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
                gt_op_preconds[action.predicate.name].add((pred_name, lit.is_negative))

    env = pddlgym.make(f"PDDLEnv{domain_name}-v0")
    actions = set()
    op_actions = set()
    ndr_actions = set()
    for o in operators:
        for lit in o.preconds.literals:
            if lit.predicate in env.action_space.predicates:
                actions.add(lit.predicate)
                op_actions.add(lit.predicate)
    for act_pred in ndrs:
        ndr_actions.add(act_pred)
        actions.add(act_pred)
 
    for action_pred in transition_data:
        actions.add(action_pred)
    # alternate color of different predicates to see better
    colors = '#1f77b4', '#ff7f0e'
    for action_pred in actions:
        fig, axs = plt.subplots(nrows=1, ncols=3)
        fig.suptitle(action_pred)

        if action_pred in op_actions:
            ops = []
            for o in operators:
                for lit in o.preconds.literals:
                    if lit.predicate.name == action_pred.name:
                        ops.append(o.pddl_str())
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axs[1].text(0,1, '\n\n'.join(ops),  fontsize=11, verticalalignment='top', bbox=props, wrap=True)
            axs[1].set_title('Operators')
        if action_pred in ndr_actions:
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axs[2].set_title('NDRs')
            axs[2].text(0, 1, '\n\n'.join([str(rule) for rule in ndrs[action_pred]]), fontsize=11, verticalalignment='top', bbox=props, wrap=True)
        if action_pred in transition_data:
            axs[0].set_title("Literals present in the initial state of non-NOPs")
            
            lit_counts = defaultdict(lambda: 0)
            total = 0
            for t in transition_data[action_pred]:
                if len(t[2]) == 0:
                    continue
                total += 1
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
            axs[0].axvline(x=total)
            axs[0].invert_yaxis()

            # Bold predicates in the preconditions of ground truth operators
            for label in axs[0].get_yticklabels():
                literal_str = label.get_text()
                pred_name = literal_str.split('(')[0]
                if (pred_name, True) in gt_op_preconds[action_pred.name]:
                    label.set_fontweight("bold")
                    label.set_color("red")
                elif (pred_name, False) in gt_op_preconds[action_pred.name]:
                    label.set_fontweight("bold")

        os.makedirs(save_path, exist_ok=True)
        plt.gcf().set_size_inches(28, 14)
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
                gt_op_preconds[action.predicate.name].add((pred_name, lit.is_negative))

    env = pddlgym.make(f"PDDLEnv{domain_name}-v0")
    actions = set()
    op_actions = set()
    ndr_actions = set()
    for o in operators:
        for lit in o.preconds.literals:
            if lit.predicate in env.action_space.predicates:
                actions.add(lit.predicate)
                op_actions.add(lit.predicate)
    for act_pred in ndrs:
        ndr_actions.add(act_pred)
        actions.add(act_pred)
 
    for action_pred in transition_data:
        actions.add(action_pred)
 
    # alternate color of different predicates to see better
    colors = '#1f77b4', '#ff7f0e'
    for action_pred in actions:
        fig, axs = plt.subplots(nrows=1, ncols=3)
        fig.suptitle(action_pred)
        if action_pred in op_actions:
            ops = []
            for o in operators:
                for lit in o.preconds.literals:
                    if lit.predicate.name == action_pred.name:
                        ops.append(o.pddl_str())
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axs[1].text(0,1, '\n\n'.join(ops),  fontsize=11, verticalalignment='top', bbox=props, wrap=True)
            axs[1].set_title('Operators')

        if action_pred in ndr_actions:

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axs[2].text(0, 1, '\n\n'.join([str(rule) for rule in ndrs[action_pred]]), fontsize=11, verticalalignment='top', bbox=props, wrap=True)
            axs[2].set_title('NDRs')

        if action_pred in transition_data:
            axs[0].set_title("Literals present in the initial state of NOPs")
            
            lit_counts = defaultdict(lambda: 0)
            total = 0
            for t in transition_data[action_pred]:
                if len(t[2]) == 0:
                    total += 1
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
            axs[0].axvline(x=total)
            axs[0].invert_yaxis()

            # Bold predicates in the preconditions of ground truth operators
            # Red is for negative
            for label in axs[0].get_yticklabels():
                literal_str = label.get_text()
                pred_name = literal_str.split('(')[0]
                if (pred_name, True) in gt_op_preconds[action_pred.name]:
                    label.set_fontweight("bold")
                    label.set_color("red")
                elif (pred_name, False)  in gt_op_preconds[action_pred.name]:
                    label.set_fontweight("bold")
               
        os.makedirs(save_path, exist_ok=True)
        plt.gcf().set_size_inches(28, 14)
        plt.savefig(os.path.join(save_path, f'{action_pred.name}-NOPs.png'), dpi=300)
        plt.close()


def view4(save_path, domain_name, curiosity_name, learning_name, seed):
    """Visualize the babbled / fallback / actions in plan with operator changes and success increases.
    plots:
    - vertical orange lines for operators changed.
    - green dots when plan is followed
    - success rate curve in black.
    """
    # Plot the success rate curve in black
    with open(os.path.join(RESULTS_PATH, domain_name, learning_name, curiosity_name, f'{domain_name}_{learning_name}_{curiosity_name}_{seed}.pkl'), 'rb') as f:
        results = pickle.load(f)
    results = np.array(results)
    xs = results[:, 0]
    ys = results[:, 1]
    plt.figure()
    plt.plot(xs, ys, color='#000000')
    # Plot green dot when following plan
    num_fallback = 0
    num_babbled = 0
    num_inplan = 0
    if "GLIB" in curiosity_name:
        with open(os.path.join(BABBLING_SOURCE_PATH, domain_name, learning_name, curiosity_name, f'{seed}_babbling_stats.pkl'), 'rb') as f:
            babbling_seq = pickle.load(f)
        following_plan_itrs = []
        following_plan_ys = []
        for itr,b in enumerate(babbling_seq):
            if not (('babbled' in b) or ('fallback' in b)):
                following_plan_itrs.append(itr)
                following_plan_ys.append(ys[itr])
                num_inplan += 1
            elif 'babbled' in b:
                num_babbled += 1
            elif 'fallback' in b:
                num_fallback += 1
        plt.scatter(following_plan_itrs, following_plan_ys, color='#008000')
    # Plot vertical orange line where operator changes
    ops_change_itrs = np.loadtxt(os.path.join(SOURCE_PATH, domain_name, learning_name, curiosity_name, seed, 'learned_ops_change_iters.txt'))
    for itr in ops_change_itrs:
        plt.axvline(x=itr, color='#FFA500')
    
    total = num_fallback + num_babbled + num_inplan
    plt.xlabel(f'Babbled: {num_babbled / total* 100}%\nFallback: {num_fallback / total* 100}%\nIn-plan: {num_inplan / total* 100}%')
    plt.gcf().set_size_inches(22, 14)
    plt.savefig(os.path.join(save_path, f'GLIB_success_plot.png'), dpi=300)
    plt.close()


def interactive_view1(domain_name, curiosity_name, learning_name, seed):
    #FIXME: not updated to show babbling/fallback/in plan actions
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
    path = os.path.join(SOURCE_PATH, domain_name, learning_name, curiosity_name, seed)
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
    with open(os.path.join(iter_path, 'operators.pkl'), 'rb') as f:
        ops = pickle.load(f)
    if not os.path.exists(filepath):
        view1(iter_save_path, ops, transition_data, domain_name)

    success_increases = np.loadtxt(os.path.join(SOURCE_PATH, domain_name, learning_name, curiosity_name, seed, 'success_increases.txt'))
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
        global curr_pos_views
        nonlocal iter_dirs

        if e.key == "right":
            curr_pos_views = curr_pos_views + 1
        elif e.key == "left":
            curr_pos_views = curr_pos_views - 1
        elif e.key == 'up':
            # next success increase
            curr_pos_views += 1
            curr_pos_views = curr_pos_views % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_views][5:])
            while curr_itr not in success_itrs:
                curr_pos_views += 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
        elif e.key == 'down':
            # prev success increase
            curr_pos_views -= 1
            curr_pos_views = curr_pos_views % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_views][5:])
            while curr_itr not in success_itrs:
                curr_pos_views -= 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
        elif e.key == 'x':
            # prev plan-following action
            if 'GLIB' not in curiosity_name:
                return
            itr = int(iter_dirs[curr_pos_views][5:]) - 1
            while (('babbled' in babbling_seq[itr]) or ('fallback' in babbling_seq[itr])) and (itr >= 0):
                itr -= 1

            if itr < 0:
                return

            iter_dir = f'iter_{itr}'
            print(iter_dir)
            if iter_dir in iter_dirs:
                curr_pos_views = iter_dirs.index(iter_dir)
            else:
                iter_dirs.append(iter_dir)
                iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
                curr_pos_views = iter_dirs.index(iter_dir)
        elif e.key == 'c':
            # next plan-following action
            if 'GLIB' not in curiosity_name:
                return
            itr = int(iter_dirs[curr_pos_views][5:]) + 1
            while (('babbled' in babbling_seq[itr]) or ('fallback' in babbling_seq[itr])) and itr <= int(iter_dirs[-1][5:]):
                itr += 1

            if itr > int(iter_dirs[-1][5:]):
                return

            iter_dir = f'iter_{itr}'
            if iter_dir in iter_dirs:
                curr_pos_views = iter_dirs.index(iter_dir)
            else:
                iter_dirs.append(iter_dir)
                iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
                curr_pos_views = iter_dirs.index(iter_dir)                   
            
        elif e.key == 'w':
            # prev op change
            curr_pos_views -= 1
            curr_pos_views = curr_pos_views % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_views][5:])
            while curr_itr not in ops_change_itrs:
                curr_pos_views -= 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
        elif e.key == 'e':
            # next op change
            curr_pos_views += 1
            curr_pos_views = curr_pos_views % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_views][5:])
            while curr_itr not in ops_change_itrs:
                curr_pos_views += 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
        elif e.key == 'r':
            # refresh
            pass
        elif e.key == 'o':
            # next episode start
            curr_pos_views += 1
            curr_pos_views = curr_pos_views % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_views][5:])
            while curr_itr not in episode_start_iters:
                curr_pos_views += 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
        elif e.key == 'i':
             # prev episode start
            curr_pos_views -= 1
            curr_pos_views = curr_pos_views % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_views][5:])
            while curr_itr not in episode_start_iters:
                curr_pos_views -= 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])           
        elif e.key == 'h':
            # prev iteration
            itr = int(iter_dirs[curr_pos_views][5:])
            if itr > 0:
                iter_dir = f'iter_{itr - 1}'
                if iter_dir not in iter_dirs:
                    iter_dirs.append(iter_dir)
                    iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
                else:
                    curr_pos_views -= 1
            else:
                return
        elif e.key == 'j':
            # next iteration
            itr = int(iter_dirs[curr_pos_views][5:])
            if itr < int(iter_dirs[-1][5:]):
                iter_dir = f'iter_{itr + 1}'
                if iter_dir not in iter_dirs:
                    iter_dirs.append(iter_dir)
                    iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
                curr_pos_views += 1
            else:
                return

        else:
            return
        curr_pos_views = curr_pos_views % len(iter_dirs)

        ax.cla()
        iter_dir = iter_dirs[curr_pos_views]
        itr_num = int(iter_dir[5:])
        iter_save_path = os.path.join(SAVE_PATH, domain_name, curiosity_name, seed, iter_dir)
        filepath = os.path.join(iter_save_path, 'nops_plot.png')
        print(filepath)
        if not os.path.exists(filepath):
            # Look ahead for the transition data, and look behind for operators and NDRs.
            curr = curr_pos_views
            transition_data_itr = None
            while curr < len(iter_dirs):
                if os.path.exists(os.path.join(path, iter_dirs[curr], 'transition_data.pkl')):
                    transition_data_itr = curr
                    break
                curr += 1
            if transition_data_itr is None:
                print("No transition data available")
                return
            curr = curr_pos_views
            ops_itr = None
            while curr >= 0:
                if os.path.exists(os.path.join(path, iter_dirs[curr], 'operators.pkl')) :
                    ops_itr = curr
                    break
                curr -= 1
            if ops_itr is None:
                print("No operators available")
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
            
            view1(iter_save_path, ops, transition_data, domain_name)

        img = mpimg.imread(filepath)
        ax.set_title(f"{iter_dir} : success rate {succ[int(iter_dir[5:])]}")
        itr = int(iter_dir[5:])
        if "GLIB" in curiosity_name:
            if not (('babbled' in babbling_seq[itr]) or ('fallback' in babbling_seq[itr])):
                goal, plan = babbling_seq[itr_num]
                ax.set_xlabel(f'goal: {goal}\nplan: {plan}\naction: {skill_seq[itr_num]}')
            else:
                ax.set_xlabel(f'{babbling_seq[itr_num]} action: {skill_seq[itr_num]}')
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
    #FIXME: not updated to show babbling/fallback/in plan actions
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
    def get_handler(skill, figax):
        """Create the event handler for this skill.

        Args:
            skill (str): name of skill
            figax (tuple): (figure, ax)
        """
        def key_event(e):
            global curr_pos_views
            global nops_view
            nonlocal iter_dirs

            if e.key == "right":
                curr_pos_views = curr_pos_views + 1
            elif e.key == "left":
                curr_pos_views = curr_pos_views - 1
            elif e.key == 'up':
                curr_pos_views += 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
                while curr_itr not in success_itrs:
                    curr_pos_views += 1
                    curr_pos_views = curr_pos_views % len(iter_dirs)
                    curr_itr = int(iter_dirs[curr_pos_views][5:])
            elif e.key == 'down':
                curr_pos_views -= 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
                while curr_itr not in success_itrs:
                    curr_pos_views -= 1
                    curr_pos_views = curr_pos_views % len(iter_dirs)
                    curr_itr = int(iter_dirs[curr_pos_views][5:])
            elif e.key == 'w':
                # prev op change
                curr_pos_views -= 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
                while curr_itr not in ops_change_itrs:
                    curr_pos_views -= 1
                    curr_pos_views = curr_pos_views % len(iter_dirs)
                    curr_itr = int(iter_dirs[curr_pos_views][5:])
            elif e.key == 'e':
                # next op change
                curr_pos_views += 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
                while curr_itr not in ops_change_itrs:
                    curr_pos_views += 1
                    curr_pos_views = curr_pos_views % len(iter_dirs)
                    curr_itr = int(iter_dirs[curr_pos_views][5:])
            elif e.key == 'r':
                # Refresh
                pass
            elif e.key == 'n':
                nops_view = 1 - nops_view
            elif e.key == 'o':
                # next episode start
                curr_pos_views += 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
                while curr_itr not in episode_start_iters:
                    curr_pos_views += 1
                    curr_pos_views = curr_pos_views % len(iter_dirs)
                    curr_itr = int(iter_dirs[curr_pos_views][5:])
            elif e.key == 'i':
                # prev episode start
                curr_pos_views -= 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
                while curr_itr not in episode_start_iters:
                    curr_pos_views -= 1
                    curr_pos_views = curr_pos_views % len(iter_dirs)
                    curr_itr = int(iter_dirs[curr_pos_views][5:])           
            elif e.key == 'h':
                # prev iteration
                itr = int(iter_dirs[curr_pos_views][5:])
                if itr > 0:
                    iter_dir = f'iter_{itr - 1}'
                    if iter_dir not in iter_dirs:
                        iter_dirs.append(iter_dir)
                        iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
                    else:
                        curr_pos_views -= 1
                else:
                    return
            elif e.key == 'j':
                # next iteration
                itr = int(iter_dirs[curr_pos_views][5:])
                if itr < int(iter_dirs[-1][5:]):
                    iter_dir = f'iter_{itr + 1}'
                    if iter_dir not in iter_dirs:
                        iter_dirs.append(iter_dir)
                        iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
                    curr_pos_views += 1
                else:
                    return
            else:
                return

            curr_pos_views = curr_pos_views % len(iter_dirs)
            iter_dir = iter_dirs[curr_pos_views]
            iter_num = int(iter_dir[5:])
            iter_save_path = os.path.join(SAVE_PATH, domain_name, curiosity_name, seed, iter_dir)
    
            curr = curr_pos_views
            transition_data_itr = None
            while curr < len(iter_dirs):
                if os.path.exists(os.path.join(path, iter_dirs[curr], 'transition_data.pkl')):
                    transition_data_itr = curr
                    break
                curr += 1
            curr = curr_pos_views
            ops_itr = None
            while curr >= 0:
                if os.path.exists(os.path.join(path, iter_dirs[curr], 'operators.pkl')) :
                    ops_itr = curr
                    break
                curr -= 1
            if ops_itr is None and transition_data_itr is None:
                return

            if transition_data_itr is not None:
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
            else:
                transition_data = {}

            with open(os.path.join(path, iter_dirs[ops_itr], 'operators.pkl'), 'rb') as f:
                ops = pickle.load(f)

            env = pddlgym.make(f'PDDLEnv{domain_name}-v0')
            actions_to_plot = set()
            for o in ops:
                for lit in o.preconds.literals:
                    if lit.predicate in env.action_space.predicates:
                        actions_to_plot.add(lit.predicate.name)
            for action in transition_data:
                actions_to_plot.add(action.name)
            create_view2 = False
            create_view3 = False
            for action in actions_to_plot:
                filepath = os.path.join(iter_save_path, f'{action}.png')
                view3_filepath = os.path.join(iter_save_path, f'{action}-NOPs.png')
                if not os.path.exists(filepath):
                    create_view2 = True
                if not os.path.exists(view3_filepath):
                    create_view3 = True

            if create_view2 or create_view3:

                with open(os.path.join(path, iter_dirs[ops_itr], 'ndrs.pkl'), 'rb') as f:
                    ndrs = pickle.load(f)
            if create_view2:
                view2(iter_save_path, ops, transition_data, ndrs, domain_name)
            if create_view3:
                view3(iter_save_path, ops, transition_data, ndrs, domain_name)

            if skill in actions_to_plot:
                fig, ax = figax
                ax.cla()
                if nops_view:
                    filepath = os.path.join(iter_save_path, f'{skill}-NOPs.png')
                else:
                    filepath = os.path.join(iter_save_path, f'{skill}.png')
                print(filepath)
                img = mpimg.imread(filepath)
                ax.set_title(f"{iter_dir} : success rate {succ[iter_num]}")
                if "GLIB" in curiosity_name:
                    if not (('babbled' in babbling_seq[iter_num]) or ('fallback' in babbling_seq[iter_num])):
                        goal, plan = babbling_seq[iter_num]
                        ax.set_xlabel(f'goal: {goal}\nplan: {plan}\naction: {skill_seq[iter_num]}')
                    else:
                        ax.set_xlabel(f'{babbling_seq[iter_num]} action: {skill_seq[iter_num]}')
                else:
                    ax.set_xlabel(f'action: {skill_seq[iter_num]}')
        
                ax.imshow(img)
                fig.canvas.draw()
        return key_event
       ###############################################################################################

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
        fig.canvas.mpl_connect('key_press_event', get_handler(act, figax))
        if act in init_actions:
            filepath = os.path.join(iter_save_path, f'{act}.png')
            img = mpimg.imread(filepath)
            ax.set_title(f'{iter_dirs[0]} : success rate {succ[0]}')
            ax.set_xlabel(f'action: {skill_seq[0]}')
            ax.imshow(img)
        else:
            ax.set_title(act)
    plt.show()

def interactive_view_123(domain_name, curiosity_name, learning_name, seed):
    """Interactive View 1, 2, and 3.

    use up/down arrow keys to toggle between success increase/decrease iterations.
    use h/j to toggle iterations.
    use 'i/o' to jump between episode starts.
    use 'w/e' to jump between operator changes.
    use 'x/c' to jump between plan-following actions.
    use 't/y' to jump between first nonNOPs.
    use left/right to scroll.
    use 'n' to change between NOPs and nonNOPs views.
    use 'p' to toggle between planning and learned operators in view1.
    use 'r' to render the screen.
     
    The view1 plot is independent of the view2 plots, which all change from one keystroke (but only rendered when press 'r').
    """

    path = os.path.join(SOURCE_PATH, domain_name, learning_name, curiosity_name, seed)
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

    success_increases = np.loadtxt(os.path.join(SOURCE_PATH, domain_name, learning_name, curiosity_name, seed, 'success_increases.txt'))
    if len(success_increases.shape) == 1:
        success_increases = success_increases[np.newaxis, :]
    success_itrs = success_increases[:, 0].tolist()

    ops_change_itrs = np.loadtxt(os.path.join(SOURCE_PATH, domain_name, learning_name, curiosity_name, seed, 'learned_ops_change_iters.txt'))
    # planning_ops_change_itrs = np.loadtxt(os.path.join(SOURCE_PATH, domain_name, learning_name, curiosity_name, seed, 'planning_ops_change_iters.txt'))

    results_path = os.path.join(RESULTS_PATH, domain_name, learning_name, curiosity_name, f'{domain_name}_{learning_name}_{curiosity_name}_{seed}.pkl')
    with open(results_path, 'rb') as f:
        results = np.array(pickle.load(f))
        succ = results[:, 1]

    episode_start_iters = np.loadtxt(os.path.join(path, 'episode_start_iters.txt'))
    first_nonNOP_iters = np.loadtxt(os.path.join(path, 'first_nonNOP_iters.txt'))

    with open(os.path.join(path, 'skill_sequence.pkl'), 'rb') as f:
        skill_seq = pickle.load(f)
    
    if "GLIB" in curiosity_name:
        with open(os.path.join(BABBLING_SOURCE_PATH, domain_name, learning_name, curiosity_name, f'{seed}_babbling_stats.pkl'), 'rb') as f:
            babbling_seq = pickle.load(f)

    def get_view2_handler(skill, figax):
        """Create the event handler for this skill.

        Args:
            skill (str): name of skill
            figax (tuple): (figure, ax)
        """
        def key_event(e):
            global curr_pos_views
            global nops_view
            nonlocal iter_dirs

            if e.key == "right":
                curr_pos_views = curr_pos_views + 1
            elif e.key == "left":
                curr_pos_views = curr_pos_views - 1
            elif e.key == 'up':
                curr_pos_views += 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
                while curr_itr not in success_itrs:
                    curr_pos_views += 1
                    curr_pos_views = curr_pos_views % len(iter_dirs)
                    curr_itr = int(iter_dirs[curr_pos_views][5:])
            elif e.key == 'down':
                curr_pos_views -= 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
                while curr_itr not in success_itrs:
                    curr_pos_views -= 1
                    curr_pos_views = curr_pos_views % len(iter_dirs)
                    curr_itr = int(iter_dirs[curr_pos_views][5:])
            elif e.key == 'w':
                # prev op change
                curr_pos_views -= 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
                while curr_itr not in ops_change_itrs:
                    curr_pos_views -= 1
                    curr_pos_views = curr_pos_views % len(iter_dirs)
                    curr_itr = int(iter_dirs[curr_pos_views][5:])
            elif e.key == 'e':
                # next op change
                curr_pos_views += 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
                while curr_itr not in ops_change_itrs:
                    curr_pos_views += 1
                    curr_pos_views = curr_pos_views % len(iter_dirs)
                    curr_itr = int(iter_dirs[curr_pos_views][5:])
            elif e.key == 'r':
                # Refresh
                pass
            elif e.key == 'n':
                nops_view = 1 - nops_view
            elif e.key == 'o':
                # next episode start
                curr_pos_views += 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
                while curr_itr not in episode_start_iters:
                    curr_pos_views += 1
                    curr_pos_views = curr_pos_views % len(iter_dirs)
                    curr_itr = int(iter_dirs[curr_pos_views][5:])
            elif e.key == 'i':
                # prev episode start
                curr_pos_views -= 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
                while curr_itr not in episode_start_iters:
                    curr_pos_views -= 1
                    curr_pos_views = curr_pos_views % len(iter_dirs)
                    curr_itr = int(iter_dirs[curr_pos_views][5:])           
            elif e.key == 'h':
                # prev iteration
                itr = int(iter_dirs[curr_pos_views][5:])
                if itr > 0:
                    iter_dir = f'iter_{itr - 1}'
                    if iter_dir not in iter_dirs:
                        iter_dirs.append(iter_dir)
                        iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
                    else:
                        curr_pos_views -= 1
                else:
                    return
            elif e.key == 'j':
                # next iteration
                itr = int(iter_dirs[curr_pos_views][5:])
                if itr < int(iter_dirs[-1][5:]):
                    iter_dir = f'iter_{itr + 1}'
                    if iter_dir not in iter_dirs:
                        iter_dirs.append(iter_dir)
                        iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
                    curr_pos_views += 1
                else:
                    return
            elif e.key == 't':
                # prev first nonNOP iter
                curr_pos_views -= 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
                while curr_itr not in first_nonNOP_iters:
                    curr_pos_views -= 1
                    curr_pos_views = curr_pos_views % len(iter_dirs)
                    curr_itr = int(iter_dirs[curr_pos_views][5:])           
            elif e.key == 'y':
                curr_pos_views += 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
                while curr_itr not in first_nonNOP_iters:
                    curr_pos_views += 1
                    curr_pos_views = curr_pos_views % len(iter_dirs)
                    curr_itr = int(iter_dirs[curr_pos_views][5:])
            else:
                return

            curr_pos_views = curr_pos_views % len(iter_dirs)
            iter_dir = iter_dirs[curr_pos_views]
            iter_num = int(iter_dir[5:])
            iter_save_path = os.path.join(SAVE_PATH, domain_name, curiosity_name, seed, iter_dir)
    
            curr = curr_pos_views
            transition_data_itr = None
            while curr < len(iter_dirs):
                if os.path.exists(os.path.join(path, iter_dirs[curr], 'transition_data.pkl')):
                    transition_data_itr = curr
                    break
                curr += 1
            curr = curr_pos_views
            ops_itr = None
            while curr >= 0:
                if os.path.exists(os.path.join(path, iter_dirs[curr], 'learned_operators.pkl')) :
                    ops_itr = curr
                    break
                curr -= 1
            if ops_itr is None and transition_data_itr is None:
                return

            if transition_data_itr is not None:
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
            else:
                transition_data = {}

            with open(os.path.join(path, iter_dirs[ops_itr], 'learned_operators.pkl'), 'rb') as f:
                ops = pickle.load(f)

            env = pddlgym.make(f'PDDLEnv{domain_name}-v0')
            actions_to_plot = set()
            for o in ops:
                for lit in o.preconds.literals:
                    if lit.predicate in env.action_space.predicates:
                        actions_to_plot.add(lit.predicate.name)
            for action in transition_data:
                actions_to_plot.add(action.name)
            create_view2 = False
            create_view3 = False
            for action in actions_to_plot:
                filepath = os.path.join(iter_save_path, f'{action}.png')
                view3_filepath = os.path.join(iter_save_path, f'{action}-NOPs.png')
                if not os.path.exists(filepath):
                    create_view2 = True
                if not os.path.exists(view3_filepath):
                    create_view3 = True

            if create_view2 or create_view3:

                with open(os.path.join(path, iter_dirs[ops_itr], 'ndrs.pkl'), 'rb') as f:
                    ndrs = pickle.load(f)
            if create_view2:
                view2(iter_save_path, ops, transition_data, ndrs, domain_name)
            if create_view3:
                view3(iter_save_path, ops, transition_data, ndrs, domain_name)

            if skill in actions_to_plot:
                fig, ax = figax
                ax.cla()
                if nops_view:
                    filepath = os.path.join(iter_save_path, f'{skill}-NOPs.png')
                else:
                    filepath = os.path.join(iter_save_path, f'{skill}.png')
                print(filepath)
                img = mpimg.imread(filepath)
                ax.set_title(f"{iter_dir} : success rate {succ[iter_num]}")
                if "GLIB" in curiosity_name:
                    if not (('babbled' in babbling_seq[iter_num]) or ('fallback' in babbling_seq[iter_num])):
                        goal, plan = babbling_seq[iter_num]
                        ax.set_xlabel(f'goal: {goal}\nplan: {plan}\naction: {skill_seq[iter_num]}')
                    else:
                        ax.set_xlabel(f'{babbling_seq[iter_num]} action: {skill_seq[iter_num]}')
                else:
                    ax.set_xlabel(f'action: {skill_seq[iter_num]}')
        
                ax.imshow(img)
                fig.canvas.draw()
        return key_event
    ###############################################################################################

    # Initialize the plots for each skill
    iter_dir = iter_dirs[0]
    iter_save_path = os.path.join(SAVE_PATH, domain_name, curiosity_name, seed, iter_dir)

    with open(os.path.join(path, iter_dirs[0], 'transition_data.pkl'),'rb')  as f:
        transition_data = pickle.load(f)
    ops_path_iter0 = os.path.join(path, iter_dirs[0], 'learned_operators.pkl')
    if os.path.exists(ops_path_iter0):
        with open(ops_path_iter0,'rb')  as f:
            ops = pickle.load(f)
    else:
        ops = []
    ndrs_path_iter0 = os.path.join(path, iter_dirs[0], 'ndrs.pkl')
    if os.path.exists(ndrs_path_iter0):
        with open(os.path.join(path, iter_dirs[0], 'ndrs.pkl'),'rb')  as f:
            ndrs = pickle.load(f)
    else:
        ndrs = {}

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
        fig.canvas.mpl_connect('key_press_event', get_view2_handler(act, figax))
        if act in init_actions:
            filepath = os.path.join(iter_save_path, f'{act}.png')
            img = mpimg.imread(filepath)
            ax.set_title(f'{iter_dirs[0]} : success rate {succ[0]}')
            ax.imshow(img)
            if 'GLIB' in curiosity_name:
                ax.set_xlabel(f'{babbling_seq[0]} action: {skill_seq[0]}')
            else:
                ax.set_xlabel(f'action: {skill_seq[0]}')
        else:
            ax.set_title(act)
 
    def key_event_view_1(e):
        global curr_pos_views
        global planning_ops_view
        nonlocal iter_dirs

        if e.key == "right":
            curr_pos_views = curr_pos_views + 1
        elif e.key == "left":
            curr_pos_views = curr_pos_views - 1
        elif e.key == 'up':
            # next success increase
            curr_pos_views += 1
            curr_pos_views = curr_pos_views % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_views][5:])
            while curr_itr not in success_itrs:
                curr_pos_views += 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
        elif e.key == 'down':
            # prev success increase
            curr_pos_views -= 1
            curr_pos_views = curr_pos_views % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_views][5:])
            while curr_itr not in success_itrs:
                curr_pos_views -= 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
        elif e.key == 'x':
            # prev plan-following action
            if 'GLIB' not in curiosity_name:
                return
            itr = int(iter_dirs[curr_pos_views][5:]) - 1
            while (('babbled' in babbling_seq[itr]) or ('fallback' in babbling_seq[itr])) and (itr >= 0):
                itr -= 1

            if itr < 0:
                return

            iter_dir = f'iter_{itr}'
            print(iter_dir)
            if iter_dir in iter_dirs:
                curr_pos_views = iter_dirs.index(iter_dir)
            else:
                iter_dirs.append(iter_dir)
                iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
                curr_pos_views = iter_dirs.index(iter_dir)
        elif e.key == 'c':
            # next plan-following action
            if 'GLIB' not in curiosity_name:
                return
            itr = int(iter_dirs[curr_pos_views][5:]) + 1
            while (('babbled' in babbling_seq[itr]) or ('fallback' in babbling_seq[itr])) and (itr >= 0) and itr <= int(iter_dirs[-1][5:]):
                itr += 1

            if itr > int(iter_dirs[-1][5:]):
                return

            iter_dir = f'iter_{itr}'
            if iter_dir in iter_dirs:
                curr_pos_views = iter_dirs.index(iter_dir)
            else:
                iter_dirs.append(iter_dir)
                iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
                curr_pos_views = iter_dirs.index(iter_dir)                   
            
        elif e.key == 'w':
            # prev op change
            curr_pos_views -= 1
            curr_pos_views = curr_pos_views % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_views][5:])
            while curr_itr not in ops_change_itrs:
                curr_pos_views -= 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
        elif e.key == 'e':
            # next op change
            curr_pos_views += 1
            curr_pos_views = curr_pos_views % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_views][5:])
            while curr_itr not in ops_change_itrs:
                curr_pos_views += 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
        elif e.key == 'r':
            # refresh
            pass
        elif e.key == 'o':
            # next episode start
            curr_pos_views += 1
            curr_pos_views = curr_pos_views % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_views][5:])
            while curr_itr not in episode_start_iters:
                curr_pos_views += 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
        elif e.key == 'i':
             # prev episode start
            curr_pos_views -= 1
            curr_pos_views = curr_pos_views % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_views][5:])
            while curr_itr not in episode_start_iters:
                curr_pos_views -= 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])           
        elif e.key == 'h':
            # prev iteration
            itr = int(iter_dirs[curr_pos_views][5:])
            if itr > 0:
                iter_dir = f'iter_{itr - 1}'
                if iter_dir not in iter_dirs:
                    iter_dirs.append(iter_dir)
                    iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
                else:
                    curr_pos_views -= 1
            else:
                return
        elif e.key == 'j':
            # next iteration
            itr = int(iter_dirs[curr_pos_views][5:])
            if itr < int(iter_dirs[-1][5:]):
                iter_dir = f'iter_{itr + 1}'
                if iter_dir not in iter_dirs:
                    iter_dirs.append(iter_dir)
                    iter_dirs = sorted(iter_dirs, key = lambda x: int(x[5:]))
                curr_pos_views += 1
            else:
                return
        elif e.key == 't':
            # prev first nonNOP iter
            curr_pos_views -= 1
            curr_pos_views = curr_pos_views % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_views][5:])
            while curr_itr not in first_nonNOP_iters:
                curr_pos_views -= 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])           
        elif e.key == 'y':
            curr_pos_views += 1
            curr_pos_views = curr_pos_views % len(iter_dirs)
            curr_itr = int(iter_dirs[curr_pos_views][5:])
            while curr_itr not in first_nonNOP_iters:
                curr_pos_views += 1
                curr_pos_views = curr_pos_views % len(iter_dirs)
                curr_itr = int(iter_dirs[curr_pos_views][5:])
        elif e.key == 'p':
            planning_ops_view = 1 - planning_ops_view
        else:
            return
        curr_pos_views = curr_pos_views % len(iter_dirs)
        # print(iter_dirs[curr_pos_views: curr_pos_views + 10], iter_dirs[curr_pos_views - 10: curr_pos_views])

        ax.cla()
        iter_dir = iter_dirs[curr_pos_views]
        itr_num = int(iter_dir[5:])
        iter_save_path = os.path.join(SAVE_PATH, domain_name, curiosity_name, seed, iter_dir)
        if planning_ops_view:
            filepath = os.path.join(iter_save_path, 'nops_plot_planning_ops.png')
        else:
            filepath = os.path.join(iter_save_path, 'nops_plot_learned_ops.png')
        print(filepath)
        if not os.path.exists(filepath):
            # Look ahead for the transition data, and look behind for operators and NDRs.
            curr = curr_pos_views
            transition_data_itr = None
            while curr < len(iter_dirs):
                if os.path.exists(os.path.join(path, iter_dirs[curr], 'transition_data.pkl')):
                    transition_data_itr = curr
                    break
                curr += 1
            if transition_data_itr is None:
                print("No transition data available")
                return
            curr = curr_pos_views
            ops_itr = None
            while curr >= 0:
                if os.path.exists(os.path.join(path, iter_dirs[curr], 'planning_operators.pkl')) :
                    ops_itr = curr
                    break
                curr -= 1
            if ops_itr is None:
                print("No operators available")
                return

            with open(os.path.join(path, iter_dirs[transition_data_itr], 'transition_data.pkl'), 'rb') as f:
                transition_data = pickle.load(f)
            # use skill sequence to create the right dataset
            action_end = int(iter_dirs[transition_data_itr][5:])
            action_start = int(iter_dir[5:])
            for action in skill_seq[action_start + 1 : action_end + 1][::-1]:
                # LIFO
                transition_data[action.predicate].pop()
            
            if planning_ops_view:
                with open(os.path.join(path, iter_dirs[ops_itr], 'planning_operators.pkl'), 'rb') as f:
                    ops = pickle.load(f)
                view1(iter_save_path, ops, transition_data, domain_name, is_learned_ops=False)
            else:
                with open(os.path.join(path, iter_dirs[ops_itr], 'learned_operators.pkl'), 'rb') as f:
                    ops = pickle.load(f)
                view1(iter_save_path, ops, transition_data, domain_name, is_learned_ops=True)


        img = mpimg.imread(filepath)
        ax.set_title(f"{iter_dir} : success rate {succ[int(iter_dir[5:])]}")
        if "GLIB" in curiosity_name:
            if not (('babbled' in babbling_seq[itr_num]) or ('fallback' in babbling_seq[itr_num])):
                goal, plan = babbling_seq[itr_num]
                ax.set_xlabel(f'goal: {goal}\nplan: {plan}\naction: {skill_seq[itr_num]}')
            else:
                ax.set_xlabel(f'{babbling_seq[itr_num]} action: {skill_seq[itr_num]}')
        else:
            ax.set_xlabel(f'action: {skill_seq[itr_num]}')

        ax.imshow(img)
        fig.canvas.draw()

    # Initialize the plot at iter=0
    iter_dir = iter_dirs[0]
    iter_path = os.path.join(path, iter_dir)
    iter_save_path = os.path.join(SAVE_PATH, domain_name, curiosity_name, seed, iter_dir)
    with open(os.path.join(iter_path, 'transition_data.pkl'), 'rb') as f:
        transition_data = pickle.load(f)
    if planning_ops_view:
        filepath = os.path.join(iter_save_path, 'nops_plot_planning_ops.png')
        if not os.path.exists(filepath):
            view1(iter_save_path, ops, transition_data, domain_name, is_learned_ops=False)
    else:
        filepath = os.path.join(iter_save_path, 'nops_plot_learned_ops.png')
        if not os.path.exists(filepath):
            view1(iter_save_path, ops, transition_data, domain_name, is_learned_ops=True)


    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', key_event_view_1)
    ax = fig.add_subplot(111)
    img = mpimg.imread(filepath)
    ax.set_title(f'{iter_dirs[0]} : success rate {succ[0]}')
    if 'GLIB' in curiosity_name:
        if not (('babbled' in babbling_seq[0]) or ('fallback' in babbling_seq[0])):
            ax.set_xlabel(f'goal: {babbling_seq[0]}\naction: {skill_seq[0]}')
        else:
            ax.set_xlabel(f'{babbling_seq[0]} action: {skill_seq[0]}')
    else:
        ax.set_xlabel(f'action: {skill_seq[0]}')
    plt.imshow(img)

    save_path = os.path.join(SAVE_PATH, domain_name, curiosity_name, seed)
    filepath = os.path.join(save_path, 'GLIB_success_plot.png')
    if not os.path.exists(filepath):
        view4(save_path, domain_name, curiosity_name, learning_name, seed)
    plt.figure()
    img = mpimg.imread(filepath)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
        
    domain_name = 'Minecraft'
    learning_name = 'LLMWarmStart+LNDR'

    # curiosity_name = 'random'
    seeds = [str(s) for s in range(162, 170)]
    
    curiosity_name = 'GLIB_G1'
    seeds = [str(s) for s in range(431, 432)]

    for seed in seeds:
        interactive_view_123(domain_name, curiosity_name, learning_name, seed)
        # view4(f'individual_plots/{domain_name}/{learning_name}/{curiosity_name}', domain_name, curiosity_name, learning_name, seed)
