from curiosity_modules import create_curiosity_module
from operator_learning_modules import create_operator_learning_module
from planning_modules import create_planning_module
from pddlgym.structs import Anti, State, Not, LiteralConjunction
from pddlgym.inference import find_satisfying_assignments
from settings import LLMConfig as lc
from openai_interface import OpenAI_Model
from settings import EnvConfig as ec
from settings import AgentConfig as ac
from ndr.learn import print_rule_set
import os
import pickle
import time
import numpy as np
from typing import Optional
import logging
from pprint import pprint


class Agent:
    """An agent interacts with an env, learns PDDL operators, and plans.
    This is a simple wrapper around three modules:
    1. a curiosity module
    2. an operator learning module
    3. a planning module
    The curiosity module selects actions to collect training data.
    The operator learning module learns operators from the training data.
    The planning module plans at test time.
    The planning module (and optionally the curiosity module) use the
    learned operators. The operator learning module contributes to them.
    """
    def __init__(self, domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path:Optional[str]):
        """

        Args:
            domain_name (str): from PDDLGym environment
            action_space : from PDDLGym environment
            observation_space : from PDDLGym environment
            curiosity_module_name (str): 
            operator_learning_name (str): 
            planning_module_name (str): 
            log_llm_path (str or None): Path to log the LLM output.
        """
        self.curiosity_time = 0.0
        self.domain_name = domain_name
        self.curiosity_module_name = curiosity_module_name
        self.operator_learning_name = operator_learning_name
        self.planning_module_name = planning_module_name

        # The main objective of the agent is to learn good operators
        self.planning_operators = set()
        self.learned_operators = set()

        self.llm = OpenAI_Model()
        self.llm_precondition_goals = dict() # Op from LLM: Op from Learner with the same action predicate (random)

        # The operator learning module learns operators. It should update the
        # agent's learned operators set
        self._operator_learning_module = create_operator_learning_module(
            operator_learning_name, self.planning_operators, self.learned_operators, self.domain_name, self.llm, self.llm_precondition_goals, log_llm_path)
        # The planning module uses the learned operators to plan at test time.
        self._planning_module = create_planning_module(
            planning_module_name, self.planning_operators, self.learned_operators, domain_name,
            action_space, observation_space)
        # The curiosity module dictates how actions are selected during training
        # It may use the learned operators to select actions
        self._curiosity_module = create_curiosity_module(
            curiosity_module_name, action_space, observation_space,
            self._planning_module, self.planning_operators, self.learned_operators,
            self._operator_learning_module, domain_name, self.llm_precondition_goals)
        
        # Flag to tell if at the episode start. Unset after observing the first effect.
        self.episode_start = False

         # Load the demos
        with open('bakingrealistic_demonstrations.pkl', 'rb') as f:
            transitions = pickle.load(f)
        self._operator_learning_module._transitions = transitions       

    ## Training time methods
    def get_action(self, state, _problem_idx):
        """Get an exploratory action to collect more training data.
           Not used for testing. Planner is used for testing."""
        if self.domain_name.lower() == 'bakingrealistic':
            obs_literals = set()
            for lit in state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    obs_literals.add(lit)
            state = State(frozenset(obs_literals), state.objects, state.goal)

        start_time = time.time()
        in_plan, op_name, action = self._curiosity_module.get_action(state)
        logging.info(f"Getting action took {time.time() - start_time}")
        self.curiosity_time += time.time()-start_time

        if in_plan:
            self._action_in_plan = op_name
        else:
            self._action_in_plan = False
        return action

    def observe(self, state, action, next_state, itr):
        """Observe a transition.

        Args:
            state (pddlgym.structs.State): initial state of the transition
            action (Literal): action taken
            effects (set[Literal]): effects of the transition
            itr (int): training iteration #
        """
        if self.domain_name.lower() == 'bakingrealistic':
            obs_literals = set()
            next_obs_literals = set()
            for lit in state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    obs_literals.add(lit)
            for lit in next_state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    next_obs_literals.add(lit)
            state = State(frozenset(obs_literals), state.objects, state.goal)
            next_state = State(frozenset(next_obs_literals), next_state.objects, next_state.goal)
        # Get effects
        effects = self._compute_effects(state, next_state)
        logging.info(f"EFFECTS: \n{effects}")
        # Add data
        self._operator_learning_module.observe(state, action, effects, start_episode=self.episode_start, itr=itr)
        # Some curiosity modules might use transition data
        start_time = time.time()
        self._curiosity_module.observe(state, action, effects)
        self.curiosity_time += time.time()-start_time
        self.episode_start = False

        # Set the info about the operator executed in the plan for the learning module.
        # If the action is not in a plan, this is None. Interested when the action is in a plan, and the operator executed has no effects (the operator fails).
        if (len(effects) == 0) and self._action_in_plan:
            self._skill_to_edit = (action.predicate, self._action_in_plan)
        else:
            self._skill_to_edit = None

    def learn(self, itr):
        # Learn
        start = time.time()
        some_learned_operator_changed, some_planning_operator_changed = self._operator_learning_module.learn(itr, skill_to_edit=self._skill_to_edit)
        # logging.info(f"Learning took {time.time() - start} s")

        # Used in LLMIterative only
        if self.operator_learning_name in ['LLM+LNDR', 'LLMIterative+LNDR']:
            self._curiosity_module.learn(itr)

        if some_learned_operator_changed:
            start_time = time.time()
            self._curiosity_module.learning_callback()
            # logging.info(f"Resetting curiosity took {time.time() - start_time}")
            self.curiosity_time += time.time()-start_time
            # for pred, dt in self._operator_learning_module.learned_dts.items():
            #     print(pred)
            #     print(dt.print_conditionals())
            # print()
        # for k, v in self._operator_learning_module._ndrs.items():
        #     print(k)
        #     print(str(v))
        return some_learned_operator_changed, some_planning_operator_changed

    def reset_episode(self, state):
        obs_literals = set()
        if self.domain_name.lower() == 'bakingrealistic':
            for lit in state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    obs_literals.add(lit)
            state = State(frozenset(obs_literals), state.objects, state.goal)

        start_time = time.time()
        self._curiosity_module.reset_episode(state)
        logging.info(f"Resetting episode for curiosity took {time.time() - start_time}")
        self.curiosity_time += time.time()-start_time
        self.episode_start = True

    @staticmethod
    def _compute_effects(state, next_state):
        positive_effects = {e for e in next_state.literals - state.literals}
        negative_effects = {Anti(ne) for ne in state.literals - next_state.literals}
        return positive_effects | negative_effects

    ## Test time methods
    def get_policy(self, problem_fname, use_learned_ops=False):
        """Get a plan given the learned operators and a PDDL problem file."""
        return self._planning_module.get_policy(problem_fname, use_learned_ops)

class InitialPlanAgent(Agent):
    """An agent with initial demonstration data to each of the 4 train tasks."""
    def __init__(self, domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path:Optional[str]):
        super().__init__(domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path)
        
        # dict: problem index -> step in the plan to execute next
        self.problem_to_plan_step = {i: 0 for i in range(len(ac.train_env.problems))}

        # dict: problem index -> list of plan steps (ground action predicate strings)
        self.plans = {}
        # self._get_plans()
        self.prev_episode_idx = None
        # Keep track of episodes that have finished at least once
        self.terminated_episodes = set()
        self.action_space = action_space
        self.obs_space = observation_space

        # Load the demos
        with open('bakingrealistic_demonstrations.pkl', 'rb') as f:
            transitions = pickle.load(f)
        self._operator_learning_module._transitions = transitions
        for action_pred in transitions:
            self._operator_learning_module._fits_all_data[action_pred] = False
        
        self.reset = False

        # Get the subgoals.
        # Keep track of the action seq to get to the last achieved subgoal.
        self.action_seq = []
        self.next_subgoal_idx = 0
        self.subgoals = []
        self._loaded_subgoals = False

        # User inputs
        self.next_action = None
        self.action_seq_reset = []
        self.observe_last_transition = False

    def reset_episode(self, state):
        if not self._loaded_subgoals:
            self._load_subgoals(state) 
            self._loaded_subgoals = True
        obs_literals = set()
        if self.domain_name.lower() == 'bakingrealistic':
            for lit in state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    obs_literals.add(lit)
            state = State(frozenset(obs_literals), state.objects, state.goal)

        start_time = time.time()
        self._curiosity_module.reset_episode(state)
        logging.info(f"Resetting episode for curiosity took {time.time() - start_time}")
        self.curiosity_time += time.time()-start_time
        self.episode_start = True


    def _load_subgoals(self, state):
        """Loads the subgoals into grounded goals."""
        SUBGOALS_TXT_PATH = '/home/catalan/GLIB-Baking-Fails-and-LLMs/realistic-baking/llm_plans/train_subgoals/problem1.txt'
        with open(SUBGOALS_TXT_PATH, 'r') as f:
            lines = f.readlines()
        subgoals = []
        for goal_line in lines:
            goal_lits = []
            for literal_str in goal_line.split(','):
                literal_str = literal_str.strip()[1:-1]
                if literal_str.startswith('not '):
                    literal_str = literal_str[len('not '):]
                    literal_str = literal_str[1:-1]
                    items = literal_str.split()
                    pred = Not(self._get_obs_predicate(items[0], items[1:], state.objects))
                else:
                    items = literal_str.split()
                    pred = self._get_obs_predicate(items[0], items[1:], state.objects)
                goal_lits.append(pred)
            #TODO: need to modify the goal for lifted mode too
            subgoals.append(LiteralConjunction(goal_lits))
        self.subgoals = subgoals
        logging.info("Loaded subgoals:")
        logging.info(self.subgoals)
    
    def _get_obs_predicate(self, pred_name:str, object_names:list, objects:frozenset):
        pred = [p for p in self.obs_space.predicates if p.name == pred_name][0]
        args = []
        for object_name in object_names:
            for o in objects:
                obj_name, _ = o._str.split(':')
                if obj_name == object_name:
                    args.append(o)
                    break
        return pred(*args)
    
    def get_action(self, state, _problem_idx):
        """Get an exploratory action to collect more training data.
           Not used for testing. Planner is used for testing."""
        if self.domain_name.lower() == 'bakingrealistic':
            obs_literals = set()
            for lit in state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    obs_literals.add(lit)
            state = State(frozenset(obs_literals), state.objects, state.goal)

        start_time = time.time()
        in_plan, op_name, action = self._curiosity_module.get_action(state, self.subgoals[self.next_subgoal_idx] if self.next_subgoal_idx < len(self.subgoals) else None)
        if not in_plan:
            # print ops, action seq, and current state
            pprint(sorted(state.literals))
            # for o in self._operator_learning_module._learned_operators:
                # logging.info(o.pddl_str())
            print_rule_set(self._operator_learning_module._ndrs)
            logging.info("Action sequence thus far")
            for act in self.action_seq:
                logging.info(act)
            self.action_seq_reset = []
            option_str = \
"""Please pick an option:

[0] Enter an action. Execute it, and observe the transition. Then, reset to the previous achieved subgoal.
[1] Enter an action sequence, and decide whether to observe the last transition. Reset to start, and then execute it. Then, try to plan to the next subgoal from there.
[2] Enter an action sequence. Reset to start, then execute it, and observe the last transition. Then, reset back to the previous subgoal.
[3] Execute a random action, observe it, and reset to the previous achieved subgoal.
[4] Execute a sequence of actions, observing all of them. Don't reset.
[5] Dump the transitions, and take a random action, observing it. Then reset to the previous achieved subgoal.
[6] Execute a sequence of actions, observing all of them. Reset to previous subgoal.
"""
            option = int(input(option_str))
            # 1. Execute the action, and observe that transition. Then, reset.
            self.option = option
            if option == 0:
                action = self._safe_action_input(state)
                self.next_action = action
            elif option == 1 or option == 2:
                action_str = input("Enter the next action, or q to quit: ")
                while action_str != 'q':
                    loop = True
                    while loop and action_str != 'q':
                        try:
                            action = self._parse_action_from_string(action_str, state.objects)
                            loop = False
                        except:
                            action_str = input("Error parsing. Re-enter the action, or enter q to quit:")
                    if action_str == 'q': break
                    self.action_seq_reset.append(action)
                    action_str = input("Enter the next action: ")
                if option == 1:
                    if input("Enter 'y' to observe the last transition (needs lowercase):") == 'y':
                        self.observe_last_transition = True
            elif option == 3:
                self.next_action = self.action_space.sample(state)
            elif option == 4:
                action_str = input("Enter the next action, or q to quit: ")
                while action_str != 'q':
                    loop = True
                    while loop and action_str != 'q':
                        try:
                            action = self._parse_action_from_string(action_str, state.objects)
                            loop = False
                        except:
                            action_str = input("Error parsing. Re-enter the action, or enter q to quit:")
                    if action_str == 'q': break
                    self.action_seq_reset.append(action)
                    action_str = input("Enter the next action: ")
            elif option == 5:
                self.next_action = self.action_space.sample(state)
                with open('transitions.pkl', 'wb') as f:
                    pickle.dump(self._operator_learning_module._transitions, f)
            elif option == 6:
                action_str = input("Enter the next action, or q to quit: ")
                while action_str != 'q':
                    loop = True
                    while loop and action_str != 'q':
                        try:
                            action = self._parse_action_from_string(action_str, state.objects)
                            loop = False
                        except:
                            action_str = input("Error parsing. Re-enter the action, or enter q to quit:")
                    if action_str == 'q': break
                    self.action_seq_reset.append(action)
                    action_str = input("Enter the next action: ")
            self._action_in_plan = False
            return None
        self.curiosity_time += time.time()-start_time

        if in_plan:
            self._action_in_plan = op_name
        else:
            self._action_in_plan = False
        return action
    
    def _safe_action_input(self, state):
        loop = True
        action_str = input("Enter the action:")
        while loop:
            try:
                action = self._parse_action_from_string(action_str, state.objects)
                loop = False
            except:
                action_str = input("Error parsing. Re-enter the action:")
        return action
    

    def observe(self, state, action, next_state, itr):
        """Observe a transition.

        Args:
            state (pddlgym.structs.State): initial state of the transition
            action (Literal): action taken
            effects (set[Literal]): effects of the transition
            itr (int): training iteration #
        """
        if self.domain_name.lower() == 'bakingrealistic':
            obs_literals = set()
            next_obs_literals = set()
            for lit in state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    obs_literals.add(lit)
            for lit in next_state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    next_obs_literals.add(lit)
            state = State(frozenset(obs_literals), state.objects, state.goal)
            next_state = State(frozenset(next_obs_literals), next_state.objects, next_state.goal)
        # Get effects
        effects = self._compute_effects(state, next_state)
        logging.info(f"EFFECTS: \n{effects}")
        # Add data
        self._operator_learning_module.observe(state, action, effects, start_episode=self.episode_start, itr=itr)
        # Some curiosity modules might use transition data
        start_time = time.time()
        self._curiosity_module.observe(state, action, effects)
        self.curiosity_time += time.time()-start_time
        self.episode_start = False

        # Check if planned to the next subgoal
        if self._action_in_plan and self.next_subgoal_idx < len(self.subgoals):
            assignments = find_satisfying_assignments(next_state.literals, self.subgoals[self.next_subgoal_idx].literals, allow_redundant_variables=False)
            if len(assignments) > 0:
                logging.info(f"ACHIEVED SUBGOAL {self.subgoals[self.next_subgoal_idx]}")
                self.next_subgoal_idx += 1
                self.action_seq.append(action)

        # Set the info about the operator executed in the plan for the learning module.
        # If the action is not in a plan, this is None. Interested when the action is in a plan, and the operator executed has no effects (the operator fails).
        if (len(effects) == 0) and self._action_in_plan:
            self._skill_to_edit = (action.predicate, self._action_in_plan)
        else:
            self._skill_to_edit = None
        
    def _parse_action_from_string(self, action_string, objects_frozenset):
        """Given action string (pred obj-0 obj-1...), parse the pddlgym action.
        """
        items = action_string.strip()[1:-1].split()
        action_predicate_name = items[0]
        object_names = items[1:]

        action_pred = [p for p in self.action_space.predicates if p.name == action_predicate_name][0]
        objects = [o for o in objects_frozenset]
        args = []
        for object_name in object_names:
            for o in objects:
                obj_name, _ = o._str.split(':')
                if obj_name == object_name:
                    args.append(o)
                    break
        return action_pred(*args)


    def learn(self, itr):
        # Learn
        start = time.time()
        some_learned_operator_changed, some_planning_operator_changed = self._operator_learning_module.learn(itr)
        # logging.info(f"Learning took {time.time() - start} s")

        # Used in LLMIterative only
        if self.operator_learning_name in ['LLM+LNDR', 'LLMIterative+LNDR']:
            self._curiosity_module.learn(itr)

        if some_learned_operator_changed:
            start_time = time.time()
            self._curiosity_module.learning_callback()
            # logging.info(f"Resetting curiosity took {time.time() - start_time}")
            self.curiosity_time += time.time()-start_time
            # for pred, dt in self._operator_learning_module.learned_dts.items():
            #     print(pred)
            #     print(dt.print_conditionals())
            # print()
        # for k, v in self._operator_learning_module._ndrs.items():
        #     print(k)
        #     print(str(v))
        return some_learned_operator_changed, some_planning_operator_changed