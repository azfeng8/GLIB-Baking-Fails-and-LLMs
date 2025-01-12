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
from settings import EnvConfig as ec
from planning_modules.base_planner import Planner, PlannerTimeoutException, \
    NoPlanFoundException
from agent import Agent
import logging
from flags import parse_flags

font = { 'size'   : 14}

matplotlib.rc('font', **font)

def learn_and_test(dataset, seed, domain_name, init_rule_sets=None):
    """evaluates the dataset on Bakinglarge and returns the successes list."""

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
                    test_env.observation_space, "GLIB_G1", "LNDR",
                    planning_module_name=ac.planner_name[domain_name])
            
    for o in ops:
        agent._planning_module._learned_operators.add(o)

        
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

def evaluate_demos(transitions_dict, seed, domain_name):
    rule_set = None
    successes, rule_set = learn_and_test(transitions_dict, seed, domain_name, rule_set)
    num_transitions = 0
    for t in transitions_dict:
        num_transitions += len(transitions_dict[t]) 

    return num_transitions, successes


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
    # Change PDDLGym domain names to domain name
    if domain_name == 'Glibdoors':
        plot_name = 'Keys and Doors'
    elif domain_name == 'Easygripper':
        plot_name = "Gripper"
    elif domain_name == "Bakinglarge":
        plot_name = "Baking-Large"
    else:
        plot_name = domain_name
    plot_path = f'results/{domain_name}/{domain_name.lower()}.png'
    os.makedirs(os.path.basename(plot_path), exist_ok=True)

    succ_rates = {plot_name: {}}
    succ_rates_std = {plot_name: {}}
    succ_rates_max_min = {plot_name: {}}

    DEMOS_PATH = f'demonstrations/{pc.domain.lower()}_demonstrations.pkl'
    with open(DEMOS_PATH, 'rb') as f:
        demos = pickle.load(f)
    total_demos, demo_successes = evaluate_demos(demos, 1, domain_name)
    print(f"Demos successes: {demo_successes}")
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

        rates = {plot_name: []}

        min_seeds = min(min_seeds, len(results_list))
        max_seeds = max(max_seeds, len(results_list))

        for results in results_list:

            # Construct these to contain the success rates arrays, one success rate per iteration
            rates_result = {plot_name: []}

            success_lists = results["successes"]

            # Assumption: The last item in the success list is from the maximum training iteration.
            i = 0
            prev_rate = {plot_name: 0}
            for itr, success_list in success_lists:
                successes = {}
                successes[plot_name] = success_list

                while i < itr:
                    rates_result[plot_name].append(prev_rate[plot_name])
                    i += 1

                rate = sum(successes[plot_name]) / len(successes[plot_name])
                rates_result[plot_name].append(rate)
                prev_rate[plot_name] = rate
 


            # extend the line here.
            if len(rates_result[plot_name]) < ac.num_train_iters[pc.domain]:
                # print(len(rates_result[plot_name]), rates_result[plot_name])
                rates_result[plot_name] = rates_result[plot_name] + (rates_result[plot_name][-1] * np.ones((ac.num_train_iters[pc.domain]- len(rates_result[plot_name]),))).tolist()
            rates[plot_name].append(rates_result[plot_name])


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
                
        # print(rates_across_seeds)
        succ_rates[plot_name][curve_name], succ_rates_std[plot_name][curve_name] = tolerant_mean(rates_across_seeds)
        succ_rates_max_min[plot_name][curve_name] = tolerant_max_min(rates_across_seeds)

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

def _main():
    # Load the demoagent and agent results
    if pc.domain != 'Bakinglarge':
        base_path = f'results_openstack/results/{pc.domain}'
    else:
        base_path = f'results_openstack/results/Bakingrealistic'
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
            if pc.domain == "Bakinglarge":
                results_path = os.path.join(base_path, learning_name, curiosity_name, f'Bakingrealistic_{learning_name}_{curiosity_name}_{agent}_{seed}.pkl')
            else:
                results_path = os.path.join(base_path, learning_name, curiosity_name, f'{pc.domain}_{learning_name}_{curiosity_name}_{agent}_{seed}.pkl')

            if os.path.exists(results_path):
                print("Loading from ", results_path)
                with open(results_path, 'rb') as f:
                    results = pickle.load(f)
                    results_list.append(results)

                all_results_filepaths.setdefault(curve_name, [])
                all_results_filepaths[curve_name].append(results_path)

            if pc.domain == "Bakinglarge":
                old_results_path = os.path.join(base_path, learning_name, curiosity_name, f'Bakingrealistic_{learning_name}_{curiosity_name}_{seed}.pkl')
            else:
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
    new_method_curve_name = "Oracle-Guidance-Demos"
    # append_demos[new_method_curve_name] = False
    #TODO: when plot student results, change this to True
    append_demos[new_method_curve_name] = True
    for seed in pc.seeds:
        if pc.domain == 'Bakinglarge':
            results_path = os.path.join(f'results/Bakingrealistic', 'LNDR', 'GLIB_G1', f'Bakingrealistic_LNDR_GLIB_G1_interactive_{seed}.pkl')
        #TODO: when plot student results, change this to GLIB_L2
        else:
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
    print("Done")

    
if __name__ == '__main__':
    # fileHandler = logging.FileHandler('out.log')
    # rootLogger = logging.getLogger()
    # rootLogger.addHandler(fileHandler)

    # consoleHandler = logging.StreamHandler()
    # rootLogger.addHandler(consoleHandler)


    # parse_flags()
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
