"""Top-level script for learning operators.
"""
from flags import parse_flags

import matplotlib
matplotlib.use("Agg")
from agent import Agent,DemonstrationsAgent, CreateDemonstrationsAgent, StudentAgent, get_input_cached
from planning_modules.base_planner import PlannerTimeoutException, \
    NoPlanFoundException
from plotting import get_plots
from settings import AgentConfig as ac
from settings import EnvConfig as ec
from settings import GeneralConfig as gc
from settings import PlottingConfig as pc
from ndr.learn import print_rule_set
from pddlgym.structs import State

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

fileHandler = logging.FileHandler('out.log')
rootLogger = logging.getLogger()
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
rootLogger.addHandler(consoleHandler)

BAKING_LARGE_TEST_CASES_DESCRIPTIONS = {
    0: "Bake 2 souffles and put them on plates",
    1: "Bake 2 cakes and put them on plates",
    2: "Bake souffle and cake, without damaging pans, putting them on plates.",
    3: "move-baked-good-in-container-to-different-container",
    4: "set-oven-with-souffle-bake-time-and-press-start",
    5: "set-oven-with-cake-bake-time-and-press-start",
    6: "fold-stiff-egg-whites-into-mixture",
    7: "pour-mixture-only",
    8: "use-stand-mixer for cake",
    9: "use-stand-mixer for souffle",
    10:"beat-egg-whites",
    11:"separate-egg-whites",
    12: "transfer-butter-from-pan-or-bowl",
    13: "transfer-egg-from-pan-or-bowl",
    14: "pour-powdery-ingredient-from-container",
    15: "remove-pan-from-oven",
    16: "put-pan-in-oven",
    17: "crack-egg",
    18: "preheat-souffle",
    19: "preheat-cake",
    20: "pour-powdery-ingredient-from-measuring-cup",
    21: "put-butter-in-container-from-measuring-cup",
}

class Runner:
    """Helper class for running experiments.
    """


    def __init__(self, agent, train_env, test_env, domain_name, curiosity_name):
        self.agent:Agent = agent
        self.train_env = train_env
        self.num_train_problems = len(self.train_env.problems)
        self.test_env = test_env
        self.domain_name = domain_name
        self.curiosity_name = curiosity_name
        self.num_train_iters = ac.num_train_iters[domain_name]

        #TODO: remove self.AUTO_EVAL in final repository
        self.AUTO_EVAL = False
        # if isinstance(agent, CreateDemonstrationsAgent):
        #     self.AUTO_EVAL = False
        # elif isinstance(agent, Agent) or isinstance(agent, DemonstrationsAgent) or isinstance(agent, StudentAgent):
        #     self.AUTO_EVAL = True
        # else:
        #     raise Exception("Not supported agent type")

    def run(self):
        """Run primitive operator learning loop.
        """
        def learn_and_test():
            nonlocal SOLVED
            # Learn and test
            if itr % ac.learning_interval[self.domain_name] == 0:

                operators_changed, _ = self.agent.learn(itr)

                if operators_changed:
                    logging.info("Operators changed.")
                    ops_change_iterations.append(itr)
                    print_rule_set(self.agent._operator_learning_module._ndrs)

                # Only rerun tests if operators have changed, or stochastic env
                if self.AUTO_EVAL and ((operators_changed or itr == 0 or \
                   itr + ac.learning_interval[self.domain_name] >= self.num_train_iters )):
                    successes_list = self._evaluate_operators(use_learned_ops=True)
                    test_solve_rate = sum(successes_list) / len(successes_list)
                    logging.info(f"Result: {test_solve_rate} solve rate")
                    if test_solve_rate == 1.0:
                        SOLVED = True
                    results["successes"].append((itr, successes_list))

                    logging.info("Learned operators:")

                    for op in sorted(self.agent.learned_operators, key=lambda op: op.name):
                        logging.info(op.pddl_str())

                else:
                    assert results, "operators_changed is False but never learned any operators..."
                    logging.debug("No operators changed, continuing...")


        problem_idx = 0 

        # Logging 
        if not self.AUTO_EVAL:
            results = {"mode": "needs_eval", "transitions": [], "ops_changed_iterations": []}
        else:
            results = {"mode": "evaluated", "successes": []} 

        ops_change_iterations = []

        episode_done = True
        # One cycle goes through all of the specified training episodes in the cycle once.
        cycle = []

        transitions = []

        itr = 0
        # Flag if experiment should end.
        SOLVED = False
        # Learn the ops from demos
        if isinstance(self.agent, StudentAgent) or isinstance(self.agent, DemonstrationsAgent):
            obs, _ = self.train_env.reset()
            self.agent.reset_episode(obs)
            self.agent.learn(0)
            logging.info("Learned operators:")
            for op in sorted(self.agent.learned_operators, key=lambda x: x.name):
                logging.info(op.pddl_str())
            learn_and_test()

        while itr < self.num_train_iters and not SOLVED:
            logging.info("Iteration {} of {}".format(itr, self.num_train_iters))

            # ask user to input which episodes to do in the next cycle
            if (isinstance(self.agent, StudentAgent) or isinstance(self.agent, CreateDemonstrationsAgent)) and len(cycle) == 0 and episode_done:
                if isinstance(self.agent, CreateDemonstrationsAgent):
                    if get_input_cached("Cycle finished. Dump transitions and exit? y or anything ") == 'y':
                        with open(f'demonstrations/{self.domain_name.lower()}_demonstrations.pkl', 'wb') as f:
                            pickle.dump(self.agent._operator_learning_module._transitions, f)
                        SOLVED = True
                        continue
 
                num_probs = len(self.train_env.problems)
                uip = get_input_cached(f"By default, all {num_probs} train problems are in the cycle. Press 'n' to enter manually the episodes, or anything else to accept.").strip()
                if uip == 'n':
                    episodes_uip = get_input_cached("Enter the episode indices, split by whitespace.").strip()
                    logging.info("Episode indices:")
                    logging.info(episodes_uip)
                    valid = True
                    accept_uip =  get_input_cached("Press y to accept").strip()
                    if not all(i < len(self.train_env.problems) for i in [int(j) for j in episodes_uip.split()]):
                        logging.info("Invalid episodes. Try again.")
                        valid = False
                    while accept_uip != 'y' or not valid:
                        episodes_uip = get_input_cached("Enter the episode indices, split by whitespace.").strip()
                        if not all(i < len(self.train_env.problems) for i in [int(j) for j in episodes_uip.split()]):
                            logging.info("Invalid episodes. Try again.")
                            valid = False
                        else:
                            valid = True
                        logging.info("Episode indices:").strip()
                        logging.info(episodes_uip)
                        accept_uip =  get_input_cached("Press y to accept").strip()
                    cycle = [int(i) for i in episodes_uip.split()]
                else:
                    cycle = list(np.random.permutation(range(num_probs)))
                logging.info(f"Episodes: " + ','.join([str(s) for s in cycle]))

                episode_done = True
            elif (not isinstance(self.agent, StudentAgent)) and len(cycle) == 0:
                num_probs = len(self.train_env.problems)
                cycle = list(np.random.permutation(range(num_probs)))

            if episode_done or ((not isinstance(self.agent, StudentAgent)) and itr % ac.max_train_episode_length[self.domain_name] == 0):
                if self.AUTO_EVAL and not isinstance(self.agent, StudentAgent):
                    problem_idx = (problem_idx + 1) % self.num_train_problems
                else:
                    problem_idx = cycle.pop(0)
                episode_done = False
                self.train_env.fix_problem_index(problem_idx)
                obs, _ = self.train_env.reset()
                logging.info(f"***********************************New episode! Problem {problem_idx}:{obs.goal}***********************************")
                self.agent.reset_episode(obs)
                if itr == 0 and isinstance(self.agent, DemonstrationsAgent):
                    self.agent.learn(0)
                    logging.info("Learned operators:")
                    for op in sorted(self.agent.learned_operators, key=lambda x: x.name):
                        logging.info(op.pddl_str())

            if isinstance(self.agent, StudentAgent) and self.agent.finished_plan:
                # Reset to previous subgoal
                self.agent.finished_plan = False
                obs, _ = self.train_env.reset()
                logging.info(f"Resetting to start state")

            logging.info("Getting action...")
            action = self.agent.get_action(obs, problem_idx, False)

            if action is None:
                if self.agent.option == 9:
                    successes_list = self._evaluate_operators(use_learned_ops=True)
                    test_solve_rate = sum(successes_list) / len(successes_list)
                    logging.info(f"Result: {test_solve_rate} solve rate")
                    self.agent.option = None
                    if test_solve_rate == 1.0:
                        SOLVED = True
                        continue
                    if sum(successes_list[:3]) > 0:
                        if get_input_cached("Solved one of the tasks of interest. End? y or anything").strip() == 'y':
                            SOLVED = True
                            continue
                elif self.agent.option == 11:
                    # End experiment.
                    SOLVED = True
                    continue
                elif self.agent.option == 12:
                    episode_done = True 
                # Clear the option.
                self.agent.option = None

            else:
                logging.info(f"Taking action {action}")
                next_obs, rew, _, _ = self.train_env.step(action)

                reset_env = self.agent.observe(obs, action, next_obs, itr)

                obs = next_obs
                learn_and_test()

                if reset_env:
                    logging.info("Resetting env")
                    obs, _ = self.train_env.reset()

                itr += 1
                transitions.append(self.agent._operator_learning_module._transitions[action.predicate][-1])

                if round(rew) == 1 and isinstance(self.agent, CreateDemonstrationsAgent):
                    episode_done = True
                #     logging.info(f"***********************************Reached goal! {obs.goal}***********************************")

        if not self.AUTO_EVAL:
            results['transitions'] = transitions
            results['ops_changed_iterations'] = ops_change_iterations
        
        return results

    def _evaluate_operators(self, use_learned_ops=True):
        """Test current operators. Return list of pass or fails (1s or 0s).
        """

        # extend the planner timeout when it's necessary in baking: evaluate in reverse order and accumulate results.
        adjusted_timeout = 300

        num_successes = 0
        num_problems = len(self.test_env.problems)

        successes = []
        success_map = {}
        if self.domain_name == 'Bakingrealistic':
            problems = range(num_problems)[::-1]
        else:
            problems = range(num_problems)
        for problem_idx in problems:
            timeout = ac.planner_timeout
            if self.domain_name == 'Bakingrealistic':
                skip = False
                if problem_idx == 0:
                    if sum([success_map[i] for i in (3, 4)]) != 2:
                        skip = True
                    else:
                        ac.planner_timeout = adjusted_timeout
                elif problem_idx == 1:
                    if sum([success_map[i] for i in (3,5)]) != 2:
                        skip = True
                    else:
                        ac.planner_timeout = adjusted_timeout
                elif problem_idx == 2:
                    if sum([success_map[i] for i in (3,4,5)]) != 3:
                        skip = True
                    else:
                        ac.planner_timeout = adjusted_timeout
                elif problem_idx == 3:
                    if sum([success_map[i] for i in (4,5)]) != 2:
                        skip = True
                    else:
                        ac.planner_timeout = adjusted_timeout
                elif problem_idx == 4:
                    if sum([success_map[i] for i in (9,16)]) != 2:
                        skip = True
                    else:
                        ac.planner_timeout = adjusted_timeout
                elif problem_idx == 5:
                    if sum([success_map[i] for i in (8, 16)]) != 2:
                        skip = True
                    else:
                        ac.planner_timeout = adjusted_timeout
                if skip:
                    successes.append(0)
                    success_map[problem_idx] = 0
                    continue

            self.test_env.fix_problem_index(problem_idx)
            obs, debug_info = self.test_env.reset()
            try:
                policy = self.agent.get_policy(debug_info["problem_file"], use_learned_ops=use_learned_ops)
            except (NoPlanFoundException, PlannerTimeoutException):
                # Automatic failure
                successes.append(0)
                success_map[problem_idx] = 0

                if self.domain_name == 'Bakingrealistic':
                    logging.info("\tTest case {}/{}, FAILED. {} successes so far. {}".format(
                    problem_idx+1, num_problems, num_successes, BAKING_LARGE_TEST_CASES_DESCRIPTIONS[problem_idx]))
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
                success_map[problem_idx] = 1
            else:
                assert reward == 0.
                successes.append(0)
                success_map[problem_idx] = 0

            if self.domain_name == 'Bakingrealistic':
                result_str = "PASSED" if reward == 1. else "FAILED"
                logging.info("\tTest case {}/{}, {}. {} successes so far. {}".format(
                problem_idx+1, num_problems, result_str, num_successes, BAKING_LARGE_TEST_CASES_DESCRIPTIONS[problem_idx]))
            else:
                logging.info("\tTest case {} of {}, {} successes so far".format(
                problem_idx+1, num_problems, num_successes))#, end="\r")
            ac.planner_timeout = timeout
 
        if self.domain_name == 'Bakingrealistic':
            successes.reverse()

        return successes

def _run_single_seed(seed, domain_name, curiosity_name, learning_name):
    start = time.time()

    ac.seed = seed
    ec.seed = seed
    np.random.seed(seed)
    ac.planner_timeout = 60 if "oracle" in curiosity_name else 10

    train_env = gym.make("PDDLEnv{}-v0".format(domain_name))
    train_env.seed(seed)
    # MAJOR HACK. Modules use the environment to access the predicates and action names.
    ac.train_env = train_env
    if gc.agent == 'use_demos':
        agent = DemonstrationsAgent(domain_name, train_env.action_space,
                    train_env.observation_space, curiosity_name, learning_name, planning_module_name=ac.planner_name[domain_name])
    elif gc.agent == 'create_demos':
         logging.info("Creating demonstrations.")
         agent = CreateDemonstrationsAgent(domain_name, train_env.action_space,
                    train_env.observation_space, curiosity_name, learning_name, planning_module_name=ac.planner_name[domain_name])

    elif gc.agent == 'student':
        agent = StudentAgent(domain_name, train_env.action_space,
                    train_env.observation_space, curiosity_name, learning_name, planning_module_name=ac.planner_name[domain_name])
       
    else:
        agent = Agent(domain_name, train_env.action_space,
                    train_env.observation_space, curiosity_name, learning_name, planning_module_name=ac.planner_name[domain_name])       

            
    test_env = gym.make("PDDLEnv{}Test-v0".format(domain_name))
    results  = Runner(agent, train_env, test_env, domain_name, curiosity_name).run()

    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results", domain_name, learning_name, curiosity_name)
    
    os.makedirs(outdir, exist_ok=True)
    cache_file = os.path.join(outdir, "{}_{}_{}_{}_{}.pkl".format(
        domain_name, learning_name, curiosity_name, agent.name, seed))
    with open(cache_file, 'wb') as f:
        pickle.dump(results, f)
        logging.info("Dumped results to {}".format(cache_file))

        
    logging.info("\n\n\nFinished single seed in {} seconds".format(time.time()-start))
    return results, cache_file


def _main():
    parse_flags()
    logger = logging.getLogger()
    logger.setLevel(gc.verbosity)

    os.makedirs(gc.results_dir, exist_ok=True)

    start = time.time()

        
    for domain_name in ec.domain_names:
        pc.domain  = domain_name
        pc.agent_learner_explorer = []
        all_results = defaultdict(list)
        append_demos_dict = {}
        all_paths_dict = defaultdict(list)
        for curiosity_name in ac.curiosity_methods_to_run:

            if gc.agent == 'use_demos':
                plot_line_name = f'{curiosity_name}-demos'
                append_demos_dict[plot_line_name] = True
                pc.agent_learner_explorer.append(('demoagent', 'LNDR', curiosity_name))
            elif gc.agent == 'create_demos':
                continue
            elif gc.agent == 'student':
                plot_line_name = f'Teacher-GLIB'
                append_demos_dict[plot_line_name] = True
                pc.agent_learner_explorer.append(('student', 'LNDR', curiosity_name))
            else:
                plot_line_name = curiosity_name
                append_demos_dict[plot_line_name] = False
                pc.agent_learner_explorer.append(('agent', 'LNDR', curiosity_name))

            for seed in range(gc.start_seed, gc.start_seed + gc.num_seeds):
                logging.info("\nRunning curiosity method: {}, with seed: {}\n".format(
                    curiosity_name, seed))

                single_seed_results, path = _run_single_seed(
                    seed, domain_name, curiosity_name, ac.learning_name)

                all_results[plot_line_name].append(single_seed_results)
                all_paths_dict[plot_line_name].append(path)
        get_plots(all_results, all_paths_dict, append_demos_dict, {}, {}, domain_name)

    logging.info("\n\n\n\n\nFinished in {} seconds".format(time.time()-start))


if __name__ == "__main__":
    _main()
