import pddlgym
import gym
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pickle
from ndr.learn import run_main_search as learn_ndrs
from settings import PlottingConfig as pc
from ndr.ndrs import NOISE_OUTCOME
from collections import defaultdict
from pddlgym.structs import LiteralConjunction
from pddlgym.parser import Operator
from pddlgym.structs import Predicate, Exists, State
from settings import AgentConfig as ac
from planning_modules.base_planner import Planner, PlannerTimeoutException, \
    NoPlanFoundException
from agent import Agent

font = {'family' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

def learn_and_test(dataset, seed, init_rule_sets=None):
    """evaluates the dataset on Bakingrealistic and returns the successes list."""
    MAX_EE_TRANSITIONS = ac.max_zpk_explain_examples_transitions[pc.domain]

    def get_batch_probs():
        assert False, 'assumed off'

    _rand_state = np.random.RandomState(seed=seed)


    rule_set = {}
    for action_predicate in dataset:

        if init_rule_sets is not None and action_predicate in init_rule_sets:
            init_rule_set = {action_predicate: init_rule_sets[action_predicate]}
        else:
            init_rule_set = None

        learned_ndrs = learn_ndrs({action_predicate : dataset[action_predicate]},
            max_timeout=ac.max_zpk_learning_time,
            max_action_batch_size=ac.max_zpk_action_batch_size[pc.domain],
            get_batch_probs=get_batch_probs,
            init_rule_sets=init_rule_set,
            rng=_rand_state,
            max_ee_transitions=MAX_EE_TRANSITIONS,
        )
        rule_set[action_predicate] = learned_ndrs[action_predicate]
    ops = []
    for act_pred in rule_set:
        name_suffix = 0
        ndrset = rule_set[act_pred]
        for ndr in ndrset.ndrs:
            # op_name = "{}{}".format(ndr.action.predicate.name, name_suffix)
            # probs, effs = ndr.effect_probs, ndr.effects
            # max_idx = np.argmax(probs)
            # max_effects = LiteralConjunction(sorted(effs[max_idx]))
            # preconds = LiteralConjunction(sorted(ndr.preconditions) + [ndr.action])
            # params = set()
            # for lit in preconds.literals + max_effects.literals:
            #     for v in lit.variables:
            #         params.add(v)
            # params= sorted(params)
            # operator = Operator(op_name, params, preconds, max_effects)
            operator = ndr.determinize(name_suffix=name_suffix)
            ops.append(operator)
            if len(operator.effects.literals) == 0 or NOISE_OUTCOME in operator.effects.literals:
                continue
            name_suffix += 1
    
    # Eval
    domain_name = pc.domain
    test_env = pddlgym.make(f"PDDLEnv{pc.domain}Test-v0")

    ac.planner_timeout = 30
    # Set these two variables to arbitrary vals to make initialization of agent not fail
    ac.seed = seed 
    ac.train_env = pddlgym.make(f"PDDLEnv{pc.domain}-v0")
    agent = Agent(domain_name, test_env.action_space,
                    test_env.observation_space, "GLIB_G1", "LNDR", log_llm_path='',
                    planning_module_name=ac.planner_name[domain_name])
            
    for o in ops:
        agent._planning_module._learned_operators.add(o)
        agent._planning_module._planning_operators.add(o)

        
    successes = []
    for i in range(len(test_env.problems)):
        test_env.fix_problem_index(i)
        obs, debug_info = test_env.reset()
        
        try:
            policy = agent.get_policy(debug_info["problem_file"], use_learned_ops=True)
        except (NoPlanFoundException,PlannerTimeoutException) as e:
            successes.append(0)
            # Automatic failure
            continue

        # Test plan open-loop
        reward = 0.
        for _ in range(40):
            try:
                action = policy(obs)
            except (NoPlanFoundException, PlannerTimeoutException):
                break
            next_obs, reward, done, _ = test_env.step(action)
            obs = next_obs
            if done:
                break

        # Reward is 1 iff goal is reached
        if reward == 1.:
            successes.append(1)
            print(f"Problem {i}: PASS")
        else:
            assert reward == 0.
            successes.append(0)
            print(f"Problem {i}: FAIL")

    return successes, rule_set

def evaluate_demos(transitions_dict, seed):
    rule_set = None
    successes, rule_set = learn_and_test(transitions_dict, seed, rule_set)
    num_transitions = 0
    for t in transitions_dict:
        num_transitions += len(transitions_dict[t]) 

    return num_transitions, successes

# BAKING_REALISTIC_TEST_CASES_DESCRIPTIONS = {
#     0: "Bake 2 souffles and put them on plates",
#     1: "Bake 2 cakes and put them on plates",
#     2: "Bake souffle and cake, without damaging pans, putting them on plates.",
#     3: "move-baked-good-in-container-to-different-container",
#     4: "set-oven-with-souffle-bake-time-and-press-start",
#     5: "set-oven-with-cake-bake-time-and-press-start",
#     6: "fold-stiff-egg-whites-into-mixture",
#     7: "pour-mixture-only",
#     8: "use-stand-mixer for cake",
#     9: "use-stand-mixer for souffle",
#     10:"beat-egg-whites",
#     11:"separate-egg-whites",
#     12: "transfer-butter-from-pan-or-bowl",
#     13: "transfer-egg-from-pan-or-bowl",
#     14: "pour-powdery-ingredient-from-container",
#     15: "remove-pan-from-oven",
#     16: "put-pan-in-oven",
#     17: "crack-egg",
#     18: "preheat-souffle",
#     19: "preheat-cake",
#     20: "pour-powdery-ingredient-from-measuring-cup",
#     21: "put-butter-in-container-from-measuring-cup",
# }

LEN_1_PLANS = set([21, 20, 19, 18, 17, 15, 14, 13, 12, 7])
DESSERT_TASKS = set([0,1,2,3,4,5])
BAKE_2_DESSERTS_TASKS = set([0,1,2])
MIXING_AND_HARDER_TASKS = set([0,1,2,3,4,5,6,8,9])
GENERALIZATION_TASKS = set([0,1])
TRAIN_TASKS = set(range(3,22))
EASY_TRAIN_TASKS = set(range(10, 22))
ALL_TASKS = set(range(22))

PLOTS = {
    # ("Success Rate on Test Tasks", 'results/Bakingrealistic/bakingrealistic_succ_generalized.png'): GENERALIZATION_TASKS,
    # ("Success Rate on Training Tasks", 'results/Bakingrealistic/bakingrealistic_succ_training.png'): TRAIN_TASKS,
    (f"{pc.domain}" if pc.domain != "Easygripper" else "Gripper", f'results/{pc.domain}/{pc.domain.lower()}_succ.png'): ALL_TASKS,
    # (f"Keys and Doors", f'results/{pc.domain}/{pc.domain.lower()}_succ.png'): ALL_TASKS,
    # (f"Baking-Large", f'results/{pc.domain}/{pc.domain.lower()}_succ.png'): ALL_TASKS,
    # ("Success Rate on Easy Training Tasks", f'results/{pc.domain}/{pc.domain.lower()}_succ_easy_training.png'): EASY_TRAIN_TASKS,


    # ("Success Rate on All Tasks (Train and Test)", 'results/Bakingrealistic/bakingrealistic_succ_demos.png'): ALL_TASKS,
}

def get_plots(results_dict, results_filepaths_dict, append_demos_dict, old_results_dict, old_results_filepaths_dict, domain_name):
    """Generates 4 plots:

    1. Success rate on all tasks
    2. Success rate on tasks testing immediately executable operators
    3. Success rate on rest of the tasks
    4. Success rate on tasks requiring baking desserts

    Average success curves are plotted.
    
    Args:
        results_dict: Dict from name of plot line to list of results dicts to plot. All dicts in the list are averaged.
        results_filepaths_dict: Dict from name of plot line to list of results PKL paths, in the same order as in results_dict.
        append_demos_dict: Dict from name of plot line to a boolean if that plot line should have demos appended.
    """
    succ_rates = {name: {} for name, _ in PLOTS}
    succ_rates_std = {name: {} for name, _ in PLOTS}
    succ_rates_max_min = {name: {} for name, _ in PLOTS}

    DEMOS_PATH = f'/home/catalan/GLIB-Baking-Fails-and-LLMs/demonstrations/{pc.domain.lower()}_demonstrations.pkl'
    with open(DEMOS_PATH, 'rb') as f:
        demos = pickle.load(f)
    total_demos, demo_successes = evaluate_demos(demos, 1)
    print(f"SAAAAAAAAAAAAAAAA: {demo_successes}")
    for curve_name, results_list in results_dict.items():
        for i,results in enumerate(results_list):
            assert results['mode'] == 'evaluated'
            if append_demos_dict[curve_name]:
                new_successes = [(total_demos - 1, demo_successes)]
                for itr, succ in results['successes']:
                    new_successes.append((itr + total_demos - 1, succ))
                results["successes"] = new_successes
                

    min_seeds = np.inf 
    max_seeds = 0
    for curve_name, results_list in results_dict.items():
        if len(results_list) == 0: 
            print(f"No results in new format found for {curve_name}")
            continue

        rates = {name: [] for name, _ in PLOTS}

        min_seeds = min(min_seeds, len(results_list))
        max_seeds = max(max_seeds, len(results_list))

        for results in results_list:

            # Construct these to contain the success rates arrays, one success rate per iteration
            rates_result = {name: [] for name, _ in PLOTS}

            success_lists = results["successes"]

            # Assumption: The last item in the success list is from the maximum training iteration.
            i = 0
            prev_rate = {name: 0 for name, _ in PLOTS}
            for itr, success_list in success_lists:
                successes = {}
                for plot_name, plot_path in PLOTS:
                    successes[plot_name] = [succ for task_index, succ in enumerate(success_list) if task_index in PLOTS[(plot_name, plot_path)]]

                while i < itr:
                    for plot_name, _ in PLOTS:
                        rates_result[plot_name].append(prev_rate[plot_name])
                    i += 1

                for plot_name, _ in PLOTS:
                    rate = sum(successes[plot_name]) / len(successes[plot_name])
                    rates_result[plot_name].append(rate)
                    prev_rate[plot_name] = rate
 


            for plot_name, _ in PLOTS:
                # extend the line here.
                if len(rates_result[plot_name]) < ac.num_train_iters[pc.domain]: #2000:
                    # print(len(rates_result[plot_name]), rates_result[plot_name])
                    rates_result[plot_name] = rates_result[plot_name] + (rates_result[plot_name][-1] * np.ones((ac.num_train_iters[pc.domain]- len(rates_result[plot_name]),))).tolist()
                rates[plot_name].append(rates_result[plot_name])


        for plot_name, _ in PLOTS:
            succ_rates[plot_name][curve_name], succ_rates_std[plot_name][curve_name], = tolerant_mean(rates[plot_name])
            succ_rates_max_min[plot_name][curve_name] = tolerant_max_min(rates[plot_name])

    for curve_name, results_list in old_results_dict.items():
        if len(results_list) == 0: 
            print(f"No results in new format found for {curve_name}")
            continue


        rates_across_seeds = []
        for results in results_list:
            rates_for_one_seed = []
            i = 0
            prev_succ_rate = 0
            for itr, succ_rate, _ in results:
                while i < itr:
                    rates_for_one_seed.append(prev_succ_rate)
                    i+= 1
                rates_for_one_seed.append(succ_rate)
                i+= 1
                prev_succ_rate = succ_rate
            rates_across_seeds.append(rates_for_one_seed)
                
        for plot_name, _ in PLOTS:
            # print(rates_across_seeds)
            succ_rates[plot_name][curve_name], succ_rates_std[plot_name][curve_name] = tolerant_mean(rates_across_seeds)
            succ_rates_max_min[plot_name][curve_name] = tolerant_max_min(rates_across_seeds)

    for plot_name, plot_path in PLOTS:
        plot_succ(plot_name, succ_rates[plot_name], succ_rates_std[plot_name], succ_rates_max_min[plot_name], plot_path, domain_name)
    

 

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    mean,std =  arr.mean(axis = -1), arr.std(axis=-1)
    return mean,std

def tolerant_max_min(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    maxes, mins = arr.max(axis=-1), arr.min(axis=-1)
    return maxes, mins

def plot_succ(title, succ_rate_dict, succ_rate_std_dict, succ_rate_max_min_dict, out_path, domain_name, plot_std=True):
    """_summary_

    Args:
        succ_rate_dict (dict): Map from curve name to list of success rates: [succ rate].
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    ax = plt.gca()
    number_of_colors = len(succ_rate_dict)
    colors = [next(ax._get_lines.prop_cycler)['color'] for _ in range(number_of_colors)]
    color_idx = 0
    for curve_name, succ_rates in sorted([(curve_name, succ_list) for curve_name, succ_list in succ_rate_dict.items()], key=lambda x: x[0]):
        xs = np.arange(len(succ_rates) + 1)[:ac.num_train_iters[domain_name]]
        results_mean = np.array([0] + succ_rates.tolist())[:ac.num_train_iters[domain_name]]
        results_std = np.array([0] + succ_rate_std_dict[curve_name].tolist())[:ac.num_train_iters[domain_name]]
        maxes, mins = succ_rate_max_min_dict[curve_name]
        maxes = np.array([0] + maxes.tolist())[:ac.num_train_iters[domain_name]]
        mins = np.array([0] + mins.tolist())[:ac.num_train_iters[domain_name]]
        plt.plot(xs, results_mean, label=curve_name, color=colors[color_idx], alpha=0.5)
        if plot_std:
            top_line = np.min(np.vstack([results_mean+results_std, maxes]),axis=0)
            bot_line = np.max(np.vstack([results_mean-results_std, mins]), axis=0)
            plt.fill_between(xs, top_line, bot_line, alpha=0.2)
        color_idx += 1
    
    plt.xlabel("Environment Interactions")
    plt.ylabel("Success Rate")
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote out to {out_path}")

def smooth_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 50))  # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
        mode='same')
    return xsmoo, ysmoo

def plot_results(domain_name, learning_name, all_results, outdir="results",
                 smooth=False, dist=False, llm_queries=None):
    """Results are lists of single-run result lists, across different
    random seeds.
    """
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), outdir)
    outfile = os.path.join(outdir, "{}_{}_{}.png".format(
        domain_name, learning_name, "dist" if dist else "succ"))
    plt.figure()
    if dist:
        ylabel = "Test Set Average Variational Distance"
    else:
        ylabel = "Test Set Success Rate"
    plt.ylabel(ylabel)

    for curiosity_module in sorted(all_results):
        results = np.array(all_results[curiosity_module])
        if len(results) == 0:
            continue
        label = curiosity_module
        xs = results[0, :, 0]
        if dist:
            ys = results[:, :, 2]
        else:
            ys = results[:, :, 1]
        results_mean = np.mean(ys, axis=0)
        # results_std = np.std(ys, axis=0)
        if smooth:
            xs, results_mean = smooth_curve(xs, results_mean)
            # _, results_std = smooth_curve(xs, results_std)
        plt.plot(xs, results_mean, label=label.replace("_", " "))
        # plt.fill_between(xs, results_mean+results_std,
        #                  results_mean-results_std, alpha=0.2)
    if llm_queries is not None:
        llm_ys = []
        llm_xs = []
        for iter, num_accept in llm_queries:
            if num_accept > 0:
                llm_ys.append(results_mean[iter])
                llm_xs.append(iter)
        plt.scatter(llm_xs, llm_ys, c='#2ca02c')

    min_seeds = min(len(x) for x in all_results.values())
    max_seeds = max(len(x) for x in all_results.values())
    if min_seeds == max_seeds:
        title = "{} Domain, {} Learner ({} seeds)".format(
            domain_name, learning_name, min_seeds)
    else:
        title = "{} Domain, {} Learner ({} to {} seeds)".format(
            domain_name, learning_name, min_seeds, max_seeds)
    if smooth:
        title += " [smoothed]"
    plt.title(title)

    plt.ylim((-0.1, 1.1))
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print("Wrote out to {}".format(outfile))

from settings import PlottingConfig as pc
def _old_main(results_path):
    """Plot the results in results/, specified by settings."""
    figures = []
    for domain, methods, seeds in zip(pc.domains, pc.methods, pc.seeds):
        lines = []
        for m,s in zip(methods, seeds):
            learning_name, curiosity_name = m
            lines.append(PlotLine(curiosity_name, learning_name, s))
        save_dir = f'plots/{domain}'
        os.makedirs(save_dir, exist_ok=True)
        figures.append(Figure(domain, lines, save_dir))
    missing_seeds = set()
    for figure in figures:
        ms = figure.run(results_path)
        missing_seeds |= ms
    print(f"Missing seeds:\n\t" + "\n\t".join(sorted(missing_seeds)))
    
import dataclasses

@dataclasses.dataclass
class PlotLine:
    def __init__(self, curiosity_method, learning_method, seeds):
        self.curiosity_method = curiosity_method
        self.learning_method = learning_method
        self.seeds = seeds
        
class Figure:
    def __init__(self, domain, plotlines:list[PlotLine], save_dir):
        self.domain = domain
        self.plotlines = plotlines
        self.save_dir = save_dir

    def run(self, results_path):
        domain = self.domain
        missing_seeds = set()
        outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), results_path)
        domain_path = os.path.join(results_path, domain)
        min_seeds = np.inf
        max_seeds = 0
        plt.figure()
        number_of_colors = len(self.plotlines)
        ax = plt.gca()
        colors = [next(ax._get_lines.prop_cycler)['color'] for _ in range(number_of_colors)]
        color_idx = 0
        for plotline in self.plotlines:
            learner = plotline.learning_method
            if learner == 'LLMWarmStart+LNDR':
                name = f'{domain}_seeds{plotline.seeds[0]}-{plotline.seeds[-1]}_{plotline.curiosity_method}_succ.png'
            explorer = plotline.curiosity_method
            seeds = plotline.seeds
            seeds_path = os.path.join(domain_path, learner, explorer)
            results = []
            min_length = np.inf
            for seed in seeds:
                pkl_fname = os.path.join(seeds_path, f'{domain}_{learner}_{explorer}_{str(seed)}.pkl')
                if not os.path.exists(pkl_fname):
                    missing_seeds.add(f"\t{domain}\t{learner}\t{explorer} Seed {seed}")
                    continue
                with open(pkl_fname, "rb") as f:
                    saved_results = pickle.load(f)
                    if len(saved_results) < min_length:
                        min_length = len(saved_results)
                results.append(saved_results)
            min_seeds = min(min_seeds, len(results))
            max_seeds = max(max_seeds, len(results))
            if len(results) == 0:
                for seed in seeds:
                    missing_seeds.add(f"\t{domain}\t{learner}\t{explorer} Seed {seed}")
                return missing_seeds
            for i,r in enumerate(results):
                results[i] = r[:min_length]
            results = np.array(results)
            label = f"{learner}, {explorer}"
            xs = results[0,:,0]
            ys = results[:, :, 1]
            results_mean = np.mean(ys, axis=0)
            std = np.std(ys, axis=0)
            std_top = results_mean + std
            std_bot = results_mean - std
            plt.plot(xs, results_mean, label=label.replace("_", " "), color=colors[color_idx])
            plt.fill_between(xs, std_bot, std_top, alpha=0.3, color=colors[color_idx])
            color_idx += 1

        if min_seeds == max_seeds:
            title = f"{domain} Domain ({min_seeds} seeds)"
        else:
            title = f"{domain} Domain, ({min_seeds} to {max_seeds} seeds)"
        
        plt.ylabel("Success rate on test problems")
        plt.title(title)
        plt.ylim((-0.1, 1.1))
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.xlabel("Iterations")

        outfile = os.path.join(self.save_dir, name)
        plt.savefig(outfile, dpi=300)
        print("Wrote out to {}".format(outfile))
        plt.close()
        return missing_seeds

def old_plotting():
    import argparse
    import shutil
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--individual_plots', action='store_true')
    parser.add_argument("-p", "--planning_results", action='store_true')
    args = parser.parse_args()

    if args.planning_results:
        path = 'results/planning_ops'
    else:
        path = 'results_openstack/results'
    llm_path = 'results/llm_iterative_log'

    if not args.individual_plots:
        _old_main(path)
    else:
    ### Make individual plots
        for domain_name in pc.domains:
            for learning_name, curiosity_name in pc.learner_explorer:
                outdir = f"individual_plots/{domain_name}/{learning_name}/{curiosity_name}"
                succ_out = f"{outdir}/succ"
                dist_out = f"{outdir}/dist"
                if os.path.exists(succ_out):
                    shutil.rmtree(succ_out)
                if os.path.exists(dist_out):
                    shutil.rmtree(dist_out)
                os.makedirs(succ_out, exist_ok=True)
                os.makedirs(dist_out, exist_ok=True)

                for seed in pc.seeds[0]:
                    all_results = defaultdict(list)
                    results_path = os.path.join(f"{path}/{domain_name}/{learning_name}/{curiosity_name}",f'{domain_name}_{learning_name}_{curiosity_name}_{seed}.pkl')
                    if not os.path.exists(results_path):
                        print(f"Missing seed {seed} for domain {domain_name} learner {learning_name} curiosity {curiosity_name}")
                        continue
                    with open(results_path, 'rb') as fh:
                        if curiosity_name == 'oracle':
                            all_results['GLIB-oracle'].append(pickle.load(fh))
                        else:
                            all_results[curiosity_name].append(pickle.load(fh))


                    llm_queries = None
                    if learning_name == 'LLM+LNDR' or learning_name == "LLMIterative+LNDR" or learning_name == "LLMIterative+ZPK":
                        p = os.path.join(llm_path, domain_name, curiosity_name, str(seed), 'experiment0', 'llm_ops_accepted.pkl') 
                        if os.path.exists(p):
                            with open(p, 'rb') as f:
                                llm_queries = pickle.load(f)
                    plot_results(f"{domain_name}{seed}", learning_name, all_results, outdir=succ_out, dist=False, llm_queries=llm_queries)
                    plot_results(f"{domain_name}{seed}", learning_name, all_results, outdir=dist_out, dist=True, llm_queries=llm_queries)

def _main():
    # Load the demoagent and agent results
    base_path = f'results_openstack/results/{pc.domain}'
    all_results = {}
    all_results_filepaths = {}
    old_result_format_results = {}
    old_results_filepaths = {}
    append_demos = {}
    for agent, learning_name, curiosity_name in pc.agent_learner_explorer:
        if agent == 'demoagent':
            if curiosity_name == 'oracle':
                curve_name = f"GLIB-oracle-demos"
            else:
                curve_name = f"{curiosity_name}-demos"
            append_demos[curve_name] = True
        else:
            if curiosity_name == 'oracle':
                curve_name = f"GLIB-oracle"
            else:
                curve_name = f"{curiosity_name}"
 
            append_demos[curve_name] = False

        results_list = []
        old_results_list = []

        for seed in pc.seeds:
            results_path = os.path.join(base_path, learning_name, curiosity_name, f'{pc.domain}_{learning_name}_{curiosity_name}_{agent}_{seed}.pkl')

            if os.path.exists(results_path):
                print("Loading from ", results_path)
                with open(results_path, 'rb') as f:
                    results = pickle.load(f)
                    results_list.append(results)

                all_results_filepaths.setdefault(curve_name, [])
                all_results_filepaths[curve_name].append(results_path)

            old_results_path = os.path.join(base_path, learning_name, curiosity_name, f'{pc.domain}_{learning_name}_{curiosity_name}_{seed}.pkl')
            if os.path.exists(old_results_path):
                print("Loading old from ", old_results_path)
                with open(old_results_path, 'rb') as f:
                    results = pickle.load(f)
                    old_results_list.append(results)

                old_results_filepaths.setdefault(curve_name, [])
                old_results_filepaths[curve_name].append(old_results_path)

        all_results[curve_name]  = results_list
        old_result_format_results[curve_name] = old_results_list

    results_list = []

    # # Load the new method results
    new_method_curve_name = "Teacher-GLIB"
    # append_demos[new_method_curve_name] = False
    #TODO: when plot student results, change this to True
    append_demos[new_method_curve_name] = True
    for seed in pc.seeds:
        # results_path = os.path.join(f'results/{pc.domain}', 'LNDR', 'GLIB_G1', f'{pc.domain}_LNDR_GLIB_G1_interactive_{seed}.pkl')
        #TODO: when plot student results, change this to GLIB_L2
        results_path = os.path.join(f'results/{pc.domain}', 'LNDR', 'GLIB_L2', f'{pc.domain}_LNDR_GLIB_L2_student_{seed}.pkl')

        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
                results_list.append(results)
                all_results_filepaths.setdefault( new_method_curve_name, [])
                all_results_filepaths[new_method_curve_name].append(results_path)
        else:
            print(f"Warning: No results found in path {results_path}..")

    all_results[new_method_curve_name] = results_list

    get_plots(all_results, all_results_filepaths, append_demos, old_result_format_results, old_results_filepaths, pc.domain)

    
if __name__ == '__main__':
    _main()

    # Evaluate demos data

    # with open(DEMO_RESULTS_PATH, 'rb') as f:
    #     demo_results = pickle.load(f)
    # demo_successes = demo_results["successes"]

    # with open(DEMO_RESULTS_PATH, 'rb') as f:
    #     results_dict = pickle.load(f)
    # results_dict['mode'] = 'evaluated'
    # # print(results_dict['successes'])
    # # s = evaluate(results_dict, 1, False)

    # # results_dict["successes"] = s
    # with open(DEMO_RESULTS_PATH, 'wb') as f:
    #     pickle.dump(results_dict, f)
