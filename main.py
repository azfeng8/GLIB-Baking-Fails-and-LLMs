"""Top-level script for learning operators.
"""
from flags import parse_flags

import matplotlib
matplotlib.use("Agg")
from agent import Agent, InitialPlanAgent, DemonstrationsAgent
from planning_modules.base_planner import PlannerTimeoutException, \
    NoPlanFoundException
from plotting import plot_results
from settings import AgentConfig as ac
from settings import EnvConfig as ec
from settings import GeneralConfig as gc
from settings import LLMConfig as lc

from collections import defaultdict

import glob
import time
from datetime import datetime
import logging
import gym
import numpy as np
import os
import pddlgym
import pickle


class Runner:
    """Helper class for running experiments.
    """


    def __init__(self, agent:Agent, train_env, test_env, domain_name, curiosity_name):
        self.agent:Agent = agent
        self.train_env = train_env
        self.num_train_problems = len(self.train_env.problems)
        self.test_env = test_env
        self.domain_name = domain_name
        self.curiosity_name = curiosity_name
        self.num_train_iters = ac.num_train_iters[domain_name]

        self.dumped_mix_cake = False
        self.dumped_mix_souffle = False

    def run(self):
        """Run primitive operator learning loop.
        """
        episode_time_step = 0
        problem_idx = 0 
        itrs_on = None

        # Logging 
        success_rates = []
        results = []
        plan_ops_results = []
        episode_start_itrs = []
        planning_ops_changed_itrs = []


        episode_time_step = 0
        prev_action = None
        episode_done = True
        # One cycle goes through all of the specified training episodes in the cycle once.
        cycle = []
        subgoals_paths = {}

        # Learn the ops from demos
        if isinstance(self.agent, InitialPlanAgent):
            self.agent.learn(0)
            logging.info("Learned operators:")
            for op in self.agent.learned_operators:
                logging.info(op.pddl_str())
            # domain_fname = self.agent._planning_module._create_domain_file(use_learned_ops=True)
            # logging.info(domain_fname)
            test_solve_rate = -1
            variational_dist = -1
            results.append((-1, test_solve_rate, variational_dist))

        for itr in range(self.num_train_iters):
            logging.info("Iteration {} of {}".format(itr, self.num_train_iters))

            # ask user to input which episodes to do in the next cycle
            if len(cycle) == 0 and episode_done:
                uip = input("Cycle finished. Dumping transitions. Filename or n to quit?")
                while not uip.endswith('.pkl') and uip != 'n':
                    uip = input("Cycle finished. Dumping transitions. Filename or n to quit?")
                if uip != 'n':
                    with open(uip.strip(), 'wb') as f:
                        pickle.dump(self.agent._operator_learning_module._transitions, f)
                logging.info(f"Dumping ops to ops.pkl...")
                with open('ops.pkl', 'wb') as f:
                    pickle.dump(self.agent.learned_operators, f)
                uip = input("Evaluate operators? y or anything")
                if uip == 'y':
                    logging.info("Evaluating operators...")
                    test_solve_rate, variational_dist, successes = self._evaluate_operators(use_learned_ops=True)
                    logging.info(f"Result: {test_solve_rate} solve rate")
                num_probs = len(self.train_env.problems)
                uip = input(f"By default, all {num_probs} train problems are in the cycle. Press 'n' to enter manually the episodes, or anything else to accept.")
                if uip == 'n':
                    episodes_uip = input("Enter the episode indices, split by whitespace.")  
                    logging.info("Episode indices:")
                    logging.info(episodes_uip)
                    valid = True
                    accept_uip =  input("Press y to accept")
                    if not all(i < len(self.train_env.problems) for i in [int(j) for j in episodes_uip.split()]):
                        logging.info("Invalid episodes. Try again.")
                        valid = False
                    while accept_uip != 'y' or not valid:
                        episodes_uip = input("Enter the episode indices, split by whitespace.")  
                        if not all(i < len(self.train_env.problems) for i in [int(j) for j in episodes_uip.split()]):
                            logging.info("Invalid episodes. Try again.")
                            valid = False
                        else:
                            valid = True
                        logging.info("Episode indices:")
                        logging.info(episodes_uip)
                        accept_uip =  input("Press y to accept")
                    cycle = [int(i) for i in episodes_uip.split()]
                else:
                    cycle = list(range(num_probs))
                logging.info(f"Episodes: " + ','.join([str(s) for s in cycle]))
                # confirm or enter subgoal paths
                DEFAULT_SUBGOALS_TXT_PATHS = [f'/home/catalan/GLIB-Baking-Fails-and-LLMs/realistic-baking/llm_plans/train_subgoals/problem{idx + 1}.txt' for idx in cycle] 
                paths_invalid = True
                while paths_invalid:
                    s = ''
                    for i, path in zip(cycle, DEFAULT_SUBGOALS_TXT_PATHS):
                        s += f'problem {i}: {path}\n'
                    s += "Confirm the above paths. Press y to accept, or anything else to enter new paths."
                    if input(s) == 'y':
                        subgoals_paths = {i: path for i, path in zip(cycle, DEFAULT_SUBGOALS_TXT_PATHS)}
                        paths_invalid = False
                    else:
                        uip = input("Enter the paths, in order of the episodes (" + ",".join([str(i) for i in cycle]) +  ") separated by white space.")
                        subgoals_paths = {i: p for i, p in zip(cycle, uip.split())}
                        if not all(os.path.exists(p) for p in subgoals_paths.values()):
                            paths_invalid = True
                        else:
                            paths_invalid = False
                episode_done = True

            if episode_done:
                problem_idx = cycle.pop(0)
                self.train_env.fix_problem_index(problem_idx)
                obs, _ = self.train_env.reset()
                logging.info(f"***********************************New episode! Problem {problem_idx}:{obs.goal}***********************************")
                self.agent.reset_episode(obs, problem_idx, subgoals_paths[problem_idx])
                episode_time_step = 0

            if self.agent.finished_preconds_plan:
                # Reset to previous subgoal
                self.agent.finished_preconds_plan = False
                obs, _ = self.train_env.reset()
                logging.info(f"Resetting to prev subgoal, executing actions:\n{self.agent.action_seq}")
                for action in self.agent.action_seq:
                    obs, rew, episode_done, _ = self.train_env.step(action)

            logging.info("Getting action...")
            action = self.agent.get_action(obs, problem_idx)
            if action is None:
                if self.agent.option == 0:
                    action = self.agent.next_action
                    next_obs, rew, episode_done, _  =  self.train_env.step(action)
                    logging.info(f"Observing action {action}")
                    self.agent.observe(obs, action, next_obs, itr)
                    obs, _ = self.train_env.reset()
                    logging.info(f"Resetting to prev subgoal, executing actions:\n{self.agent.action_seq}")
                    next_obs = obs
                    for action in self.agent.action_seq:
                        next_obs, rew, episode_done, _ = self.train_env.step(action)
                elif self.agent.option == 1:
                    obs, _ = self.train_env.reset()
                    logging.info(f"Resetting to start, and executing actions:\n{self.agent.action_seq_reset}")
                    for i, action in enumerate(self.agent.action_seq_reset):
                        next_obs, rew, episode_done, _ = self.train_env.step(action)
                        if i == len(self.agent.action_seq_reset) - 1 and self.agent.observe_last_transition:
                            logging.info(f"Observing action {action}")
                            self.agent.observe(obs, action, next_obs, itr)
                        obs = next_obs
                elif self.agent.option == 2:
                    obs, _ = self.train_env.reset()
                    logging.info(f"Resetting to start, and executing actions:\n{self.agent.action_seq_reset}. Then resetting to prev subgoal")
                    for i, action in enumerate(self.agent.action_seq_reset):
                        next_obs, rew, episode_done, _ = self.train_env.step(action)
                        if i == len(self.agent.action_seq_reset) - 1 and self.agent.observe_last_transition:
                            logging.info(f"Observing action {action}")
                            self.agent.observe(obs, action, next_obs, itr)
                        obs = next_obs                   
                    obs, _ = self.train_env.reset()
                    next_obs = obs
                    for action in self.agent.action_seq:
                        next_obs, rew, episode_done, _ = self.train_env.step(action)
                elif self.agent.option == 3 or self.agent.option == 5:
                    action = self.agent.next_action
                    logging.info(f"Observing action {action}")
                    next_obs, rew, episode_done, _ = self.train_env.step(action)
                    self.agent.observe(obs, action, next_obs, itr)
                    obs, _ = self.train_env.reset()
                    next_obs = obs
                    for action in self.agent.action_seq:
                        next_obs, rew, episode_done, _ = self.train_env.step(action)
                elif self.agent.option == 4:
                    for action in self.agent.action_seq_reset:
                        next_obs, rew, episode_done, _ = self.train_env.step(action) 
                        self.agent.observe(obs, action, next_obs, itr)
                        obs = next_obs
                        itr += 1
                elif self.agent.option == 6:
                    for action in self.agent.action_seq_reset:
                        next_obs, rew, episode_done, _ = self.train_env.step(action) 
                        self.agent.observe(obs, action, next_obs, itr)
                        obs = next_obs
                        itr += 1                   
                    obs, _ = self.train_env.reset()
                    next_obs = obs
                    for action in self.agent.action_seq:
                        next_obs, rew, episode_done, _ = self.train_env.step(action)
 
            else:
                logging.info(f"Taking action {action}")
                next_obs, rew, episode_done, _ = self.train_env.step(action)
                if prev_action == action:
                    # logging.info(f"Obs:\n{obs} \n\nNext obs:\n{next_obs}")
                    if input("Dump transitions? y/n") == 'y':
                        with open(f'transitions.pkl', 'wb') as f:
                            pickle.dump(self.agent._operator_learning_module._transitions, f)
                if round(rew) == 1: logging.info(f"***********************************Reached goal! {obs.goal}***********************************")
                self.agent.observe(obs, action, next_obs, itr)

            obs = next_obs
            prev_action = action
            episode_time_step += 1

            if (episode_time_step == 1) and ('LNDR' in self.agent.operator_learning_name):
                episode_start_itrs.append(itr)

            # Learn and test
            if itr % ac.learning_interval[self.domain_name] == 0:

                if self.domain_name == "PybulletBlocks" and self.curiosity_name == "oracle":
                    operators_changed = True
                else:
                    operators_changed, planning_operators_changed = self.agent.learn(itr)

                # Only rerun tests if operators have changed, or stochastic env
                if (operators_changed or ac.planner_name[self.domain_name] == "ffreplan" or \
                   itr + ac.learning_interval[self.domain_name] >= self.num_train_iters):
                    if operators_changed:
                        logging.info("Operators changed.")
                    logging.info("Learned operators:")
                    for op in self.agent.learned_operators:
                        logging.info(op.pddl_str())

                    test_solve_rate = -1
                    variational_dist = -1


                else:
                    # assert results, "operators_changed is False but never learned any operators..."
                    logging.debug("No operators changed, continuing...")

                    test_solve_rate = results[-1][1]
                    variational_dist = results[-1][2]
                    logging.info(f"Result: {test_solve_rate} {variational_dist}")


                if planning_operators_changed or \
                   itr + ac.learning_interval[self.domain_name] >= self.num_train_iters:
                    planning_ops_changed_itrs.append(itr)
 
                results.append((itr, test_solve_rate, variational_dist))

        if itrs_on is None:
            itrs_on = self.num_train_iters
        curiosity_avg_time = self.agent.curiosity_time/itrs_on
        
        return results, curiosity_avg_time, plan_ops_results

    def _evaluate_operators(self, use_learned_ops=True):
        """Test current operators. Return (solve rate on test suite,
        average variational distance).
        """
        BAKING_REALISTIC_TEST_CASES_DESCRIPTIONS = {
            0: "Bake souffle and cake, without damaging pans, putting them on plates.",
            1: "move-baked-good-in-container-to-different-container",
            2: "set-oven-with-souffle-bake-time-and-press-start",
            3: "set-oven-with-cake-bake-time-and-press-start",
            4: "fold-stiff-egg-whites-into-mixture",
            5: "pour-mixture-only",
            6: "use-stand-mixer for cake",
            7: "use-stand-mixer for souffle",
            8: "beat-egg-whites",
            9: "separate-egg-whites",
            10: "transfer-butter-from-pan-or-bowl",
            11: "transfer-egg-from-pan-or-bowl",
            12: "pour-powdery-ingredient-from-container",
            13: "remove-pan-from-oven",
            14: "put-pan-in-oven",
            15: "crack-egg",
            16: "preheat-souffle",
            17: "preheat-cake",
            18: "pour-powdery-ingredient-from-measuring-cup",
            19: "put-butter-in-container-from-measuring-cup",
        }
        if self.domain_name == "PybulletBlocks" and self.curiosity_name == "oracle":
            # Disable oracle for pybullet.
            return 0.0, 1.0
        num_successes = 0
        if self.domain_name in ec.num_test_problems:
            num_problems = ec.num_test_problems[self.domain_name]
        else:
            num_problems = len(self.test_env.problems)
        successes = []
        passed_cases = set()
        if self.domain_name.lower() == 'bakingrealistic':
            problems = range(num_problems-1, -1, -1)
        else:
            problems = range(num_problems)
        for problem_idx in problems:
            #FIXME: First get the operator learner to learn mixing. That is the bottleneck for the harder tasks.
            if self.domain_name.lower() == 'bakingrealistic':
                if (problem_idx == 6) and (
                    # Problem 6 needs these cases to pass
                    18 not in passed_cases
                    or 19 not in passed_cases 
                    or 15 not in passed_cases
                ):
                    continue
                if (problem_idx == 7) and (
                    # Problem 7 needs these cases to pass
                    9 not in passed_cases
                    or 18 not in passed_cases
                    or 19 not in passed_cases 
                    or 15 not in passed_cases
                ):
                    continue
            self.test_env.fix_problem_index(problem_idx)
            obs, debug_info = self.test_env.reset()
            try:
                policy = self.agent.get_policy(debug_info["problem_file"], use_learned_ops=use_learned_ops)
            except (NoPlanFoundException, PlannerTimeoutException):
                # Automatic failure
                successes.append(0)
                if self.domain_name.lower() == 'bakingrealistic':
                    logging.info("\tTest case {}/{}, FAILED. {} successes so far. {}".format(
                    problem_idx+1, num_problems, num_successes, BAKING_REALISTIC_TEST_CASES_DESCRIPTIONS[problem_idx]))
                else:
                    logging.info("\tTest case {} of {}, {} successes so far".format(
                    problem_idx+1, num_problems, num_successes))
                continue
            # Test plan open-loop
            reward = 0.
            for _ in range(ac.max_test_episode_length[self.domain_name]):
                try:
                    action = policy(obs)
                except (NoPlanFoundException, PlannerTimeoutException):
                    break
                obs, reward, done, _ = self.test_env.step(action)
                if done:
                    break

            # Reward is 1 iff goal is reached
            if reward == 1.:
                num_successes += 1
                successes.append(1)
                passed_cases.add(problem_idx)
            else:
                assert reward == 0.
                successes.append(0)

            if self.domain_name.lower() == 'bakingrealistic':
                result_str = "PASSED" if reward == 1. else "FAILED"
                logging.info("\tTest case {}/{}, {}. {} successes so far. {}".format(
                problem_idx+1, num_problems, result_str, num_successes, BAKING_REALISTIC_TEST_CASES_DESCRIPTIONS[problem_idx]))
            else:
                logging.info("\tTest case {} of {}, {} successes so far".format(
                problem_idx+1, num_problems, num_successes))#, end="\r")
 
        variational_dist = 0
        if 6 in passed_cases and not self.dumped_mix_cake: 
            with open('mix-for-cake.pkl', 'wb') as f:
                pickle.dump(self.agent._operator_learning_module._transitions, f)
            self.dumped_mix_cake = True
        if 7 in passed_cases and not self.dumped_mix_souffle:
            with open('mix-for-souffle.pkl', 'wb') as f:
                pickle.dump(self.agent._operator_learning_module._transitions, f)
            self.dumped_mix_souffle = True
 

        return float(num_successes)/num_problems, variational_dist, successes

def _run_single_seed(seed, domain_name, curiosity_name, learning_name, log_llmi_path:str):
    start = time.time()

    ac.seed = seed
    ec.seed = seed
    ac.planner_timeout = 60 if "oracle" in curiosity_name else 400 

    train_env = gym.make("PDDLEnv{}-v0".format(domain_name))
    train_env.seed(seed)
    # MAJOR HACK. Only used by oracle_curiosity.py and by the LLM-based
    # learner, which uses the environment to access the predicates and
    # action names.
    ac.train_env = train_env
    agent = InitialPlanAgent(domain_name, train_env.action_space,
                train_env.observation_space, curiosity_name, learning_name, log_llm_path=log_llmi_path,
                planning_module_name=ac.planner_name[domain_name])
        
    test_env = gym.make("PDDLEnv{}Test-v0".format(domain_name))
    results, curiosity_avg_time, plan_ops_results  = Runner(agent, train_env, test_env, domain_name, curiosity_name).run()
    with open("results/timings/{}_{}_{}_{}.txt".format(domain_name, curiosity_name, learning_name, seed), "w") as f:
        f.write("{} {} {} {} {}\n".format(domain_name, curiosity_name, learning_name, seed, curiosity_avg_time))

    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results", domain_name, learning_name, curiosity_name)
    plan_ops_outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results", 'planning_ops', domain_name, learning_name, curiosity_name)
    
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(plan_ops_outdir, exist_ok=True)
    cache_file = os.path.join(outdir, "{}_{}_{}_{}.pkl".format(
        domain_name, learning_name, curiosity_name, seed))
    with open(cache_file, 'wb') as f:
        pickle.dump(results, f)
        logging.info("Dumped results to {}".format(cache_file))
    with open(os.path.join(plan_ops_outdir, "{}_{}_{}_{}.pkl".format(
        domain_name, learning_name, curiosity_name, seed)), 'wb') as f:
        pickle.dump(plan_ops_results, f)

    if gc.dataset_logging:
        if "GLIB" in curiosity_name:
            path = os.path.join(f'results', 'GLIB', domain_name, learning_name, curiosity_name)
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, f'{seed}_babbling_stats.pkl'), 'wb') as f:
                pickle.dump(agent._curiosity_module.line_stats, f)
            if "LLM" in curiosity_name:
                with open(os.path.join(path, f'{seed}_llm_babbling_stats.pkl') ,'wb') as f:
                    pickle.dump(agent._curiosity_module.llm_line_stats, f)

        
    logging.info("\n\n\nFinished single seed in {} seconds".format(time.time()-start))
    return {curiosity_name: results}


def _main():
    parse_flags()
    logger = logging.getLogger()
    logger.setLevel(gc.verbosity)

    os.makedirs(gc.results_dir, exist_ok=True)
    os.makedirs(gc.timings_dir, exist_ok=True)
    os.makedirs(gc.vardisttrans_dir, exist_ok=True)

    start = time.time()

    for domain_name in ec.domain_name:
        all_results = defaultdict(list)
        for curiosity_name in ac.curiosity_methods_to_run:
            for seed in range(gc.start_seed, gc.start_seed + gc.num_seeds):
                logging.info("\nRunning curiosity method: {}, with seed: {}\n".format(
                    curiosity_name, seed))

                if lc.iterative_log_path:
                    llm_iterative_log_path = os.path.join(lc.iterative_log_path, domain_name, curiosity_name, str(seed))
                else:
                    llm_iterative_log_path = None

                single_seed_results = _run_single_seed(
                    seed, domain_name, curiosity_name, ac.learning_name, llm_iterative_log_path)
                for cur_name, results in single_seed_results.items():
                    all_results[cur_name].append(results)
                plot_results(domain_name, ac.learning_name, all_results)
                plot_results(domain_name, ac.learning_name, all_results, dist=True)

        plot_results(domain_name, ac.learning_name, all_results)
        plot_results(domain_name, ac.learning_name, all_results, dist=True)

    logging.info("\n\n\n\n\nFinished in {} seconds".format(time.time()-start))


if __name__ == "__main__":
    _main()
