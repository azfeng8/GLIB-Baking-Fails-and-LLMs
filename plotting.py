import pddlgym
import gym
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pickle
from ndr.learn import run_main_search as learn_ndrs
from settings import AgentConfig as ac
from ndr.ndrs import NOISE_OUTCOME
from collections import defaultdict
from pddlgym.structs import LiteralConjunction
from pddlgym.parser import Operator
from pddlgym.structs import Predicate, Exists, State
from settings import AgentConfig as ac
from planning_modules.base_planner import Planner, PlannerTimeoutException, \
    NoPlanFoundException
from agent import Agent


def learn_and_test(dataset):
    """evaluates the dataset on Bakingrealistic and returns the successes list."""
    MAX_EE_TRANSITIONS = ac.max_zpk_explain_examples_transitions['Bakingrealistic']

    def get_batch_probs():
        assert False, 'assumed off'

    init_rule_sets = None
    _rand_state = np.random.RandomState(seed=1)


    rule_set = {}
    for action_predicate in dataset:
        learned_ndrs = learn_ndrs({action_predicate : dataset[action_predicate]},
            max_timeout=ac.max_zpk_learning_time,
            max_action_batch_size=ac.max_zpk_action_batch_size['Bakingrealistic'],
            get_batch_probs=get_batch_probs,
            init_rule_sets=init_rule_sets,
            rng=_rand_state,
            max_ee_transitions=MAX_EE_TRANSITIONS,
        )
        rule_set[action_predicate] = learned_ndrs[action_predicate]
    ops = []
    for act_pred in rule_set:
        name_suffix = 0
        ndrset = rule_set[act_pred]
        for ndr in ndrset.ndrs:
            op_name = "{}{}".format(ndr.action.predicate.name, name_suffix)
            indices = [i for i, eff in enumerate(ndr.effects) if len(eff) > 0 ]
            effs = ndr.effects
            ops = []
            for idx in indices:
                op_name = "{}{}".format(ndr.action.predicate.name, name_suffix)
                effects = LiteralConjunction(sorted(effs[idx]))
                if len(effects.literals) == 0 or NOISE_OUTCOME in effects.literals:
                    continue
                preconds = LiteralConjunction(sorted(ndr.preconditions) + [ndr.action])
                params = set()
                for lit in preconds.literals + effects.literals:
                    for v in lit.variables:
                        params.add(v)
                params= sorted(params)
                ops.append(Operator(op_name, params, preconds, effects))
                name_suffix += 1
    
    # Eval
    domain_name = 'Bakingrealistic'

    test_env = pddlgym.make("PDDLEnvBakingrealisticTest-v0")

    ac.planner_timeout = 400
    # Set these two variables to arbitrary vals to make initialization of agent not fail
    ac.seed = 1
    ac.train_env = pddlgym.make("PDDLEnvBakingrealistic-v0")
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
        else:
            assert reward == 0.
            successes.append(0)

    return successes

def evaluate(results_dict):
    """Learns the operators and outputs success arrays at each of the iterations where operators changed."""
    assert results_dict['mode'] == 'needs_eval'
    success_lists = [] # (itr, success list)
    transitions = results_dict['transitions']
    iterations_to_eval = results_dict['ops_changed_iterations']

    # Always evaluate the last one
    last_idx = len(transitions) - 1
    if last_idx not in iterations_to_eval:
        iterations_to_eval.append(last_idx)
    
    dataset = {} 
    for i, t in enumerate(transitions):
        dataset.setdefault(t[1].predicate, [])
        dataset[t[1].predicate].append(t)
        if i in iterations_to_eval:
            successes = learn_and_test(dataset)
            success_lists.append((i, successes))

    return success_lists

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

LEN_1_PLANS = set([21, 20, 19, 18, 17, 16])
DESSERT_TASKS = set([0,1,2,3,4,5])
MIXING_AND_HARDER_TASKS = set([0,1,2,3,4,5,6,7,8,9])

def get_plots_for_bakinglarge(results_dict, results_filepaths_dict):
    """Generates 4 plots:

    1. Success rate on all tasks
    2. Success rate on tasks testing immediately executable operators
    3. Success rate on rest of the tasks
    4. Success rate on tasks requiring baking desserts

    Average success curves are plotted.
    
    Args:
        results_dict: Dict from name of plot line to list of results dicts to plot. All dicts in the list are averaged.
        results_filepaths_dict: Dict from name of plot line to list of results PKL paths, in the same order as in results_dict.
    """
    succ_rate_all_tasks = {}
    succ_rate_length1_plans = {}
    succ_rate_mixing_and_harder_tasks = {}
    succ_rate_baking_desserts = {}

    for curve_name, results_list in results_dict.items():
        for i,results in enumerate(results_list):
            if results['mode'] == 'needs_eval':
                print(f"Evaluating for curve_name, {i}th result...")
                success_lists = evaluate(results)
                results['mode'] = 'evaluated'
                results['successes'] = success_lists
                print("Dumping success lists from evaluated transitions...")
                with open(results_filepaths_dict[curve_name][i], 'wb') as f:
                    pickle.dump(results, f)

    min_seeds = np.inf 
    max_seeds = 0
    for curve_name, results_list in results_dict.items():
        assert len(results_list) > 0, f"No results found"

        all_tasks_rates = []
        mixing_and_harder_rates = []
        len1_plans_rates = []
        baking_desserts_rates = []

        min_seeds = min(min_seeds, len(results_list))
        max_seeds = max(max_seeds, len(results_list))

        for results in results_list:

            # Construct these to contain the success rates arrays, one success rate per iteration
            all_tasks_rates_result = []
            mixing_and_harder_rates_result = []
            len1_plans_rates_result = []
            baking_desserts_rates_result = []

            success_lists = results["successes"]

            # Assumption: The last item in the success list is from the maximum training iteration.
            i = 0
            prev_all_tasks_rate = 0
            prev_mixing_and_harder_rate = 0
            prev_len1_plans_rate = 0
            prev_baking_desserts_rate = 0
            for itr, success_list in success_lists:

                mixing_and_harder = [succ for task_index, succ in enumerate(success_list) if task_index in MIXING_AND_HARDER_TASKS]
                len1_plans = [succ for task_index, succ in enumerate(success_list) if task_index in LEN_1_PLANS]
                baking_desserts = [succ for task_index, succ in enumerate(success_list) if task_index in DESSERT_TASKS]

                while i < itr:
                    all_tasks_rates_result.append(prev_all_tasks_rate)
                    mixing_and_harder_rates_result.append(prev_mixing_and_harder_rate)
                    len1_plans_rates_result.append(prev_len1_plans_rate)
                    baking_desserts_rates_result.append(prev_baking_desserts_rate)
                    i += 1

                all_tasks_rate = sum(success_list) / len(success_list)
                mixing_and_harder_rate = sum(mixing_and_harder)/len(mixing_and_harder)
                len1_plans_rate = sum(len1_plans)/len(len1_plans)
                baking_desserts_rate = sum(baking_desserts)/len(baking_desserts)

                all_tasks_rates_result.append(all_tasks_rate)
                mixing_and_harder_rates_result.append(mixing_and_harder_rate)
                len1_plans_rates_result.append(len1_plans_rate)
                baking_desserts_rates_result.append(baking_desserts_rate)

                prev_len1_plans_rate = len1_plans_rate
                prev_mixing_and_harder_rate = mixing_and_harder_rate
                prev_all_tasks_rate = all_tasks_rate
                prev_baking_desserts_rate = baking_desserts_rate

            all_tasks_rates.append(all_tasks_rates_result)
            mixing_and_harder_rates.append(mixing_and_harder_rates_result)
            len1_plans_rates.append(len1_plans_rates_result)
            baking_desserts_rates.append(baking_desserts_rates_result)

        # Truncate the success rate array lengths to the minimum length one
        if not all(len(s) == len(all_tasks_rates[0]) for s in all_tasks_rates):
            min_length = min(len(s) for s in all_tasks_rates)
            print(f"Not all seeds are the same length! Truncating to length {min_length}...")
            all_tasks_rates = [s[:min_length] for s in all_tasks_rates]
            len1_plans_rates = [s[:min_length] for s in len1_plans_rates]
            mixing_and_harder_rates = [s[:min_length] for s in mixing_and_harder_rates]       
            baking_desserts_rates = [s[:min_length] for s in baking_desserts_rates]

        succ_rate_all_tasks[curve_name] = np.mean(all_tasks_rates, axis=0)
        succ_rate_length1_plans[curve_name] = np.mean(len1_plans_rates, axis=0)
        succ_rate_mixing_and_harder_tasks[curve_name] = np.mean(mixing_and_harder_rates, axis=0)
        succ_rate_baking_desserts[curve_name] = np.mean(baking_desserts_rates, axis=0)

    if min_seeds != max_seeds:
        plot_succ(f"Success Rate on All Tasks ({min_seeds} to {max_seeds} seeds)", succ_rate_all_tasks, 'results/Bakingrealistic/bakingrealistic_succ_all_tasks.png')
        plot_succ(f"Success Rate on Length 1 Plan Tasks ({min_seeds} to {max_seeds} seeds)", succ_rate_length1_plans, 'results/Bakingrealistic/bakingrealistic_succ_len1.png')   
        plot_succ(f"Success Rate on Mixing and Subsequent Tasks ({min_seeds} to {max_seeds} seeds)", succ_rate_mixing_and_harder_tasks, 'results/Bakingrealistic/bakingrealistic_succ_mixing_and_harder.png')
        plot_succ(f"Success Rate on Baking Dessert Tasks ({min_seeds} to {max_seeds} seeds)", succ_rate_baking_desserts, 'results/Bakingrealistic/bakingrealistic_succ_baking_desserts.png')
    else:
        plot_succ(f"Success Rate on All Tasks ({min_seeds} seeds)", succ_rate_all_tasks, 'results/Bakingrealistic/bakingrealistic_succ_all_tasks.png')
        plot_succ(f"Success Rate on Length 1 Plan Tasks ({min_seeds} seeds)", succ_rate_length1_plans, 'results/Bakingrealistic/bakingrealistic_succ_len1.png')   
        plot_succ(f"Success Rate on Mixing and Subsequent Tasks ({min_seeds} seeds)", succ_rate_mixing_and_harder_tasks, 'results/Bakingrealistic/bakingrealistic_succ_mixing_and_harder.png')
        plot_succ(f"Success Rate on Baking Dessert Tasks ({min_seeds} seeds)", succ_rate_baking_desserts, 'results/Bakingrealistic/bakingrealistic_succ_baking_desserts.png')

 

def plot_succ(title, succ_rate_dict, out_path):
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
        plt.plot(np.arange(len(succ_rates)), succ_rates, label=curve_name, color=colors[color_idx])
        color_idx += 1
    
    plt.xlabel("Iterations")
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
def main(results_path):
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
        main(path)
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
                        all_results[curiosity_name].append(pickle.load(fh))


                    llm_queries = None
                    if learning_name == 'LLM+LNDR' or learning_name == "LLMIterative+LNDR" or learning_name == "LLMIterative+ZPK":
                        p = os.path.join(llm_path, domain_name, curiosity_name, str(seed), 'experiment0', 'llm_ops_accepted.pkl') 
                        if os.path.exists(p):
                            with open(p, 'rb') as f:
                                llm_queries = pickle.load(f)
                    plot_results(f"{domain_name}{seed}", learning_name, all_results, outdir=succ_out, dist=False, llm_queries=llm_queries)
                    plot_results(f"{domain_name}{seed}", learning_name, all_results, outdir=dist_out, dist=True, llm_queries=llm_queries)

if __name__ == '__main__':
    base_path = 'results/Bakingrealistic'
    all_results = {}
    all_results_filepaths = {}
    for learning_name, curiosity_name in pc.learner_explorer:
        results_list = []
        for seed in pc.seeds:
            results_path = os.path.join(base_path, learning_name, curiosity_name, f'Bakingrealistic_{learning_name}_{curiosity_name}_agent_{seed}.pkl')

            if os.path.exists(results_path):
                with open(results_path, 'rb') as f:
                    results = pickle.load(f)
                    results_list.append(results)

                all_results_filepaths.setdefault(f"{curiosity_name}", [])
                all_results_filepaths[f"{curiosity_name}"].append(results_path)

        all_results[f"{curiosity_name}"]  = results_list

    results_list = []

    new_method_curve_name = f"Method: Demonstrations + Curriculum + Learned Precondition as Goals"
    for seed in pc.seeds:
        results_path = os.path.join(base_path, 'LNDR', 'GLIB_G1', f'Bakingrealistic_LNDR_GLIB_G1_interactive_{seed}.pkl')

        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
                results_list.append(results)
                all_results_filepaths.setdefault( new_method_curve_name, [])
                all_results_filepaths[new_method_curve_name].append(results_path)
        else:
            print(f"Warning: No results found in path {results_path}..")


    all_results[new_method_curve_name] = results_list

    get_plots_for_bakinglarge(all_results, all_results_filepaths)

    