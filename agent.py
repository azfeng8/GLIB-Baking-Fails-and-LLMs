from curiosity_modules import create_curiosity_module
from operator_learning_modules import create_operator_learning_module
from planning_modules import create_planning_module
from pddlgym.structs import Anti, State
from settings import LLMConfig as lc
from openai_interface import OpenAI_Model
from settings import EnvConfig as ec
from settings import AgentConfig as ac
import os
import pickle
import time
import numpy as np
from typing import Optional
import logging


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
        

    ## Training time methods
    def get_action(self, state, _problem_idx):
        """Get an exploratory action to collect more training data.
           Not used for testing. Planner is used for testing."""
        start_time = time.time()
        in_plan, op_name, action = self._curiosity_module.get_action(state)
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
        some_learned_operator_changed, some_planning_operator_changed = self._operator_learning_module.learn(itr, skill_to_edit=self._skill_to_edit)

        # Used in LLMIterative only
        if self.operator_learning_name in ['LLM+LNDR', 'LLMIterative+LNDR']:
            self._curiosity_module.learn(itr)

        if some_learned_operator_changed:
            start_time = time.time()
            self._curiosity_module.learning_callback()
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
        start_time = time.time()
        self._curiosity_module.reset_episode(state)
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
        self._get_plans()
        self.prev_episode_idx = None
        # Keep track of episodes that have finished at least once
        self.terminated_episodes = set()
        self.action_space = action_space
 
    def get_action(self, state, problem_idx):

        if self.prev_episode_idx is not None and self.prev_episode_idx != problem_idx:
            self.terminated_episodes.add(self.prev_episode_idx)

        # If this is the first time in this episode, execute the plan until the episode terminates or the plan terminates.
        if problem_idx not in self.terminated_episodes and self.problem_to_plan_step[problem_idx] != "DONE":
            plan = self.plans[problem_idx]
            plan_step = self.problem_to_plan_step[problem_idx]
            action =  self._parse_action_from_string(plan[plan_step], state.objects)
            if plan_step + 2 > len(plan):
                self.problem_to_plan_step[problem_idx] = "DONE"
            else:
                self.problem_to_plan_step[problem_idx] += 1
            self.prev_episode_idx = problem_idx
            self._action_in_plan = False
            return action
            
        self.prev_episode_idx = problem_idx

        in_plan, op_name, action = self._curiosity_module.get_action(state)

        if in_plan:
            self._action_in_plan = op_name
        else:
            self._action_in_plan = False
        return action

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

    def _get_plans(self):
        """Fill in self.plans with the plans from txt files."""
        FILEPATHS = [
           '/home/catalan/GLIB-Baking-Fails-and-LLMs/realistic-baking/llm_plans/train/problem1.txt',
           '/home/catalan/GLIB-Baking-Fails-and-LLMs/realistic-baking/llm_plans/train/problem2.txt',
           '/home/catalan/GLIB-Baking-Fails-and-LLMs/realistic-baking/llm_plans/train/problem3.txt',
           '/home/catalan/GLIB-Baking-Fails-and-LLMs/realistic-baking/llm_plans/train/problem4.txt',
        ]
        for problem_i, filepath in enumerate(FILEPATHS):
            with open(filepath, 'r') as f:
                self.plans[problem_i] = [l for l in f.readlines() if l.strip() != '']