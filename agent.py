"""DONE: only evaluate when prompted to, until get 1 seed (non continuous) completed.
DONE: Make the runs deterministic: sets into np.random.permutation over lists
TODO: make restarts/stops read/write to the same results pkl: do this once get one start/stop done.
"""
import math
import traceback 
from planning_modules.base_planner import PlannerTimeoutException, \
    NoPlanFoundException
from collections import defaultdict
import itertools
from curiosity_modules import create_curiosity_module
from operator_learning_modules import create_operator_learning_module
from planning_modules import create_planning_module
from pddlgym.structs import Anti, State, Not, LiteralConjunction, ground_literal, Exists, Literal, Type, TypedEntity, Predicate
from pddlgym.parser import Operator, PDDLDomainParser
from settings import LLMConfig as lc
from openai_interface import OpenAI_Model
from settings import EnvConfig as ec
from settings import AgentConfig as ac
from ndr.learn import print_rule_set
from ndr.ndrs import NOISE_OUTCOME
from llm_parsing import GoalParser, LLM_PDDL_Parser
import os
import pickle
import time
import numpy as np
from typing import Optional, Tuple
import logging
from pprint import pprint
from copy import deepcopy


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
        self.name = "agent"
        self.curiosity_time = 0.0
        self.domain_name = domain_name
        self.curiosity_module_name = curiosity_module_name
        self.operator_learning_name = operator_learning_name
        self.planning_module_name = planning_module_name
        self._rand_state = np.random.RandomState(seed=ac.seed)
        # The main objective of the agent is to learn good operators
        self.planning_operators = set()
        self.learned_operators = set()

        self.llm = OpenAI_Model()
        self.llm_precondition_goals = dict() # Op from LLM: Op from Learner with the same action predicate (random)

        # The operator learning module learns operators. It should update the
        # agent's learned operators set
        self._operator_learning_module = create_operator_learning_module(
            operator_learning_name, self.planning_operators, self.learned_operators, self.domain_name, self.llm, self.llm_precondition_goals, log_llm_path, self._rand_state)
        # The planning module uses the learned operators to plan at test time.
        self._planning_module = create_planning_module(
            planning_module_name, self.planning_operators, self.learned_operators, domain_name,
            action_space, observation_space)
        # The curiosity module dictates how actions are selected during training
        # It may use the learned operators to select actions
        self._curiosity_module = create_curiosity_module(
            curiosity_module_name, action_space, observation_space,
            self._planning_module, self.planning_operators, self.learned_operators,
            self._operator_learning_module, domain_name, self.llm_precondition_goals, self._rand_state)
        
        # Flag to tell if at the episode start. Unset after observing the first effect.
        self.episode_start = False


    ## Training time methods
    def get_action(self, state, _problem_idx, _precond_targeting_only):
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
        # if (len(effects) == 0) and self._action_in_plan:
        #     self._skill_to_edit = (action.predicate, self._action_in_plan)
        # else:
        #     self._skill_to_edit = None

    def learn(self, itr):
        # Learn
        start = time.time()
        some_learned_operator_changed, some_planning_operator_changed = self._operator_learning_module.learn(itr, skill_to_edit=None) #FIXME skill_to_edit is not able to run like this
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

    def reset_episode(self, state, _):
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

class InteractiveAgentGrounded(Agent):
    """An agent with initial demonstration data to each of the 4 train tasks.

    Must be run with GLIB_G, since goals will be grounded.
    """
    def __init__(self, domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path:Optional[str]):
        super().__init__(domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path)
        
        self.name = 'interactive'
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

        self._plan_to_op_preconds_failed = False

        # Load the demos
        with open(f'demonstrations/{self.domain_name.lower()}_demonstrations.pkl', 'rb') as f:
            transitions = pickle.load(f)
        self._operator_learning_module._transitions = transitions
        # for action_pred in transitions:
        #     self._operator_learning_module._fits_all_data[action_pred] = True
 
        for action_pred in transitions:
            self._operator_learning_module._fits_all_data[action_pred] = False
        
        # Get the subgoals.
        # Keep track of the action seq to get to the last achieved subgoal.
        self.action_seq = []
        self._plan_to_next_subgoal = None
        self._last_plan_to_next_subgoal = None
        self.next_subgoal_idx = 0
        self.subgoals = []
        self._loaded_subgoals = False
        # Keep track if the last action was part of a plan to subgoals.
        self._action_in_plan = False
        # Keep track of executed actions since the last subgoal.
        self.actions_since_last_subgoal = []

        # User inputs
        self.next_action = None
        # Keep track of the actions from the user input
        self.action_seq_reset = []
        self.observe_last_transition = False

        # Planning to preconditions
        self.precondition_targeting = True
        self._preconds_plan = None
        # This is set by the Runner.run() method and also self._get_action_with_preconds_as_goals()
        self.finished_preconds_plan = False
        self._last_preconds_action = None
        self._ops_preconds_executed = set()

        # Keeps track of preconditions already planned to from states
        self._visited_preconds_states = {a: set() for a in action_space.predicates} # Map from action predciate to set

        # # Load visited set
        # with open("ops_visited.pkl", 'rb') as f:
        #     self._ops_preconds_executed = pickle.load(f)
        # with open('visited_preconds.pkl', 'rb') as f:
        #     self._visited_preconds_states = pickle.load(f)
        # with open('rand_state.pkl', 'rb') as f:
        #     self._rand_state.set_state(pickle.load(f))

        # # # Load NDRs
        # with open('ndrs.pkl', 'rb') as f:
        #     self._operator_learning_module._ndrs = pickle.load(f)


        # # Load ops
        # with open('ops.pkl', 'rb') as f:
        #     ops = pickle.load(f)
        # for op in ops:
        #     self.learned_operators.add(op)
        #     self.planning_operators.add(op)


    def reset_episode(self, state, subgoals_path):
        if subgoals_path:
            self._load_subgoals(state, subgoals_path) 
        else:
            self.subgoals = None
        self.next_subgoal_idx = 0
        self.action_seq = []
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


    def _load_subgoals(self, state, subgoals_file):
        """Loads the subgoals into grounded goals."""
        with open(subgoals_file, 'r') as f:
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
            subgoals.append(LiteralConjunction(goal_lits))
        self.subgoals = subgoals
        self.next_subgoal_idx = 0
        self.actions_since_last_subgoal = []
        logging.info(f"Loaded subgoals from {subgoals_file}: ")
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
    
    def _get_action_with_preconds_as_goals(self, state, ops_to_exclude):
        """Returns the action, or None, if the stopping condition is reached.

        Args:
            state: the current state
            ops_to_exclude: don't do precondition targeting for these actions.

        Stopping condition:
            If all of the preconditions are either unreachable from this state or the same preconditions has already had an action tried from it.
        """
        # Have successfully executed the plan to the operator preconds, and will execute the operator next
        if self._preconds_plan is not None and len(self._preconds_plan) == 1:
            self.finished_preconds_plan = True
            self.actions_since_last_subgoal = []
            logging.info(f"FOLLOWING PLAN: {self._preconds_plan}")
            self._ops_preconds_executed.add(self._op_preconds_to_execute)
            self._op_preconds_to_execute = None
            return self._preconds_plan.pop()

        # Follow plan to the operator's preconditions
        elif self._preconds_plan is not None and len(self._preconds_plan) > 0:
            self.finished_preconds_plan = False
            logging.info(f"FOLLOWING PLAN: {self._preconds_plan}")
            return self._preconds_plan.pop(0)

        # set a hyperparameter for how many ground preconditions to try.
        NUM_TRIES = 200

        action_predicates = set(p.name for p in self.action_space.predicates)
        for op in self._rand_state.permutation(sorted(self.learned_operators, key=lambda op: op.name)):
            # since the last time operators were learned, if operator has been successfully executed at the end of the plan, or
            # the plan failed in the middle to the operator preconditions, skip it.
            if op.name in self._ops_preconds_executed:
                continue
            if op.name in ops_to_exclude: continue
            preconds = op.preconds.literals

            logging.info(f"Trying preconds for op: {op.name}: {preconds}")
            ground_preconds_list = self._get_ground_preconds(op, state)
            for i in self._rand_state.permutation(len(ground_preconds_list))[:NUM_TRIES]:
                grounded_precond = ground_preconds_list[i]
                preconds_hash = get_hashable_preconds_action(grounded_precond)
                ground_act = [p for p in grounded_precond if p.predicate.name in action_predicates][0]
                if (preconds_hash, state) in self._visited_preconds_states[ground_act.predicate]:
                    continue
                grounded_precond_no_act = [p for p in grounded_precond if p.predicate.name not in action_predicates]
                plan = self._get_plan_to_preconds(grounded_precond_no_act, state)
                self._visited_preconds_states[ground_act.predicate].add((preconds_hash, state))
                if plan == 'skip':
                    self._ops_preconds_executed.add(op.name)
                    break
                elif plan is not None:
                    self._preconds_plan = plan + [ground_act]
                    logging.info(f"Found plan to preconds: {preconds_hash}")
                    logging.info(f"PLAN: {self._preconds_plan}")
                    self._op_preconds_to_execute = op.name
                    if len(self._preconds_plan) == 1:
                        self.finished_preconds_plan = True
                        self.actions_since_last_subgoal = []
                        self._ops_preconds_executed.add(self._op_preconds_to_execute)
                        self._op_preconds_to_execute = None
                    return self._preconds_plan.pop(0)

        # once done, proceed to the next subgoal in the file.
        return None
    
    def _get_ground_preconds(self, operator, state):
        """Return a list of lists of grounded literals that form the precondition.

        Returns:
        [[ grounded precond literals version 1], [precond grounding version 2], ...]
        """
        preconds = sorted(operator.preconds.literals)
        assignments = self._get_assignments(preconds, state)
        ground_preconds_list = []
        for assignment in assignments:
            ground_preconds = tuple(ground_literal(p, assignment) for p in preconds)
            ground_preconds_list.append(ground_preconds)
        return ground_preconds_list
    
    def _get_assignments(self, precond_literals, state):
        """Return a list of assignments of parameter variable (TypedEntity) to object (TypedEntity)."""
        objects = sorted(state.objects)
        var_names_to_type = {}

        for lit in precond_literals:
            for var in lit.variables:
                t = var._str.split(':')[1]
                var_names_to_type[var] = t.strip()

        var_names_types = sorted([(v, t) for v,t in var_names_to_type.items()], key=lambda x: x[0])

        def recurse(var_names_types, i,  objects, assignment, assignments=[]):
            if i == len(var_names_types):
                assignments.append(deepcopy(assignment))
                return assignments
            var, t = var_names_types[i]
            for obj in objects:
                if obj._str.split(':')[-1] == t and obj not in assignment.values():
                    assignment[var] = obj
                    # recurse
                    assignments = recurse(var_names_types, i+1, objects, assignment, assignments)
                    del assignment[var]
            return assignments
        assignments = recurse(var_names_types, 0, objects, {}, [])
        return assignments

    def _get_plan_to_preconds(self, grounded_precond_lits:list, state):
        """Returns None if no plan found, otherwise a list of action literals."""
        goal = LiteralConjunction(grounded_precond_lits)

        # Create a pddl problem file with the goal and current state
        problem_fname = self._curiosity_module._create_problem_pddl(
            state, goal, prefix='glibg1_preconds')

        # logging.info(problem_fname)
        # Get a plan
        try:
            plan, _ = self._planning_module.get_plan(
                problem_fname, use_cache=False, use_learned_ops=True)
            os.remove(problem_fname)
            return plan
        except NoPlanFoundException:
            logging.info(f"No plan found.")
        except PlannerTimeoutException:
            logging.info(f"PLANNER TIMED OUT")
            if input("skip this preconditions? y or anything").strip() == 'y':
                os.remove(problem_fname)
                return 'skip'

        os.remove(problem_fname)

        return None
        

    def get_action(self, state, _problem_idx, precond_targeting_only: bool):
        """Get an exploratory action to collect more training data.
           Not used for testing. Planner is used for testing.
        
        Args:
            state: The current state.
            precond_targeting_only: True if mode is to target preconditions.
        """
        if self.domain_name.lower() == 'bakingrealistic':
            obs_literals = set()
            for lit in state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    obs_literals.add(lit)
            state = State(frozenset(obs_literals), state.objects, state.goal)

        # Before getting to a new subgoal, try out all the operator preconditions if they haven't been tried before, to refine incorrect preconditions.
        if self.precondition_targeting:
            # prompt if want to target preconditions or not.
            if not ac.auto_target_preconds and not precond_targeting_only:
                target_preconds = (input(f"Target preconditions (y) or skip precondition targeting? Ops that would be tried: {[o.name for o in self.learned_operators if o.name not in self._ops_preconds_executed]}\n y or anything").strip() == 'y')
                ops_to_exclude = set()
                # select operator names that should be skipped.
                operator_names = set(o.name for o in self.learned_operators)
                if not target_preconds:
                    uip = input("Enter an op name to exclude, or n to quit: ").strip()
                    while uip != 'n':
                        if uip in  operator_names:
                            ops_to_exclude.add(uip)
                        else:
                            logging.info(f"Invalid operator name: {uip}")
                        uip = input("Enter an op name to exclude, or n to quit: ").strip()

            else:
                target_preconds = True 
                ops_to_exclude = set()
                
            if target_preconds:
                self._action_in_plan = False
                logging.info("Getting plan to precondition...")
                action = self._get_action_with_preconds_as_goals(state, ops_to_exclude)
                if action is None:
                    self._action_in_plan_to_preconds = False
                    self.precondition_targeting = False 
                else:
                    self._action_in_plan_to_preconds = True
                    return action
            else:
                self.precondition_targeting = False
                self._action_in_plan_to_preconds = False
        else:
            self._action_in_plan_to_preconds = False


        if precond_targeting_only:
            return None

        if self.next_subgoal_idx == len(self.subgoals):
            return self._prompt_demos_or_subgoals(state)

        logging.info("Getting plan to next subgoal...")
        if self._plan_to_next_subgoal is not None and len(self._plan_to_next_subgoal) > 0:
            logging.info(f"Continuing plan: {self._plan_to_next_subgoal}")
            act = self._plan_to_next_subgoal.pop(0)
            # add to the actions list
            self.actions_since_last_subgoal.append(act)
            return act 
        # Get a plan to the next subgoal
        problem_fname = self._curiosity_module._create_problem_pddl(state, self.subgoals[self.next_subgoal_idx], prefix='glibg1_subgoal')
        plan = None
        try:
            plan, _ = self._planning_module.get_plan(
                problem_fname, use_cache=False, use_learned_ops=True)
            os.remove(problem_fname)
        except NoPlanFoundException:
            logging.info(f"No plan found.")
        except PlannerTimeoutException:
            logging.info(f"Planner timed out.")
        if plan:
            logging.info(f"Found plan: {plan}")
            self._action_in_plan = True
            if self._last_plan_to_next_subgoal == plan:
                logging.info("REPEATED PLAN")
                self._plan_to_next_subgoal = None
                return self._prompt_demos_or_subgoals(state)
            else:
                self._last_plan_to_next_subgoal = plan

            self._plan_to_next_subgoal = plan
            act = self._plan_to_next_subgoal.pop(0)
            # add to the actions list
            self.actions_since_last_subgoal.append(act)
            return act
        else:
            return self._prompt_demos_or_subgoals(state)

    def _prompt_demos_or_subgoals(self, state):
        # print ops, action seq, and current state
        pprint(sorted(state.literals))
        # for o in self._operator_learning_module._learned_operators:
            # logging.info(o.pddl_str())
        print_rule_set(self._operator_learning_module._ndrs)
        logging.info("Action sequence to previous subgoal below. Look at logs to see all the actions from last subgoal to now")
        for act in self.action_seq:
            logging.info(act)
        self.action_seq_reset = []
        if self.next_subgoal_idx < len(self.subgoals):
            logging.info(f"Next subgoal to achieve: {self.subgoals[self.next_subgoal_idx]}")
        else:
            logging.info("No more subgoals to achieve.")
        option_str = \
"""Please pick an option:

*** Demonstration ***
[0] Enter an action. Execute it, and observe the transition. Then, reset to the previous achieved subgoal.
[2] Enter an action sequence. Reset to start, then execute it, and observing all the transitions. Then, reset back to the previous subgoal.
[6] Enter an action sequence. Execute it from here, observing all of them. Reset to previous subgoal.
[4] Execute a sequence of actions from here, observing all of them. Don't reset.

*** Curriculum ***
[7] Abandon this subgoals list, and enter a new curriculum to execute from this state.
[8] Abandon this subgoals list, reset the episode, and try the new curriculum.

*** Precondition Targeting ***
[10] Reset episode and target preconditions in the selected episode.

*** Utils ***
[5] Dump the transitions and operators.
[9] Evaluate operators.
[11] End experiment.

*** MISC ***
[3] Execute a random action, observe it, and reset to the previous achieved subgoal.
"""
        try:
            option = int(input(option_str))
        except:
            option = None
        while option is None and option not in [0,2,3,4,5,6,7,8,9,10,11]:
            try:
                option = int(input(option_str))
            except:
                option = None           
        # 1. Execute the action, and observe that transition. Then, reset.
        self.option = option
        if option == 0:
            action = self._safe_action_input(state)
            self.next_action = action
        #when resetting to the previous subgoal, clear the actions list.
            self.actions_since_last_subgoal = []
        elif option == 2:
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
        #when resetting to the previous subgoal, clear the actions list.
            self.actions_since_last_subgoal = []
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
                # when not resetting to the previous subgoal (option 4) add to the actions buffer
                self.actions_since_last_subgoal.append(action)
                action_str = input("Enter the next action: ")
        elif option == 5:
            self.next_action = self.action_space.sample(state)
            dump_intermediate_state(self)
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
        #when resetting to the previous subgoal, clear the actions list.
            self.actions_since_last_subgoal = []
        elif option == 7:
            logging.info(f"Starting new curriculum from current state.")
            subgoals_list_fname = input("Enter the new subgoals list: ")
            while not os.path.exists(subgoals_list_fname):
                subgoals_list_fname = input("Enter the new subgoals list: ")               
            self._load_subgoals(state, subgoals_list_fname)
        elif option == 8:           
            logging.info(f"Starting new curriculum from start of episode.")
            subgoals_list_fname = input("Enter the new subgoals list: ")
            while not os.path.exists(subgoals_list_fname):
                subgoals_list_fname = input("Enter the new subgoals list: ")               
            self._load_subgoals(state, subgoals_list_fname)
            self.action_seq = []
            self.action_seq_reset = []

        self._action_in_plan = False
        return None
    
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

        # Check if planned to preconditions
        if self._action_in_plan_to_preconds:
            # Stop executing the plan if it failed in the middle.
            if len(effects) == 0:
                self.finished_preconds_plan = True
                # About to reset to the previous subgoal, so clear this list.
                self.actions_since_last_subgoal = []
                self._preconds_plan = None
                # If the operators don't change as a result of adding the NOP, then the op preconds should be added to the visited set.
                self._plan_to_op_preconds_failed = True
            else:
                self._plan_to_op_preconds_failed = False
                
        else:
            if len(effects) == 0:
                self._plan_to_next_subgoal = None

        # Check if planned to the next subgoal
        if self._action_in_plan and self.next_subgoal_idx < len(self.subgoals):
            # for each Not(), assert that the positive version isn't in the literals, and call the helper only on the positive literals in the subgoal
            lits = self.subgoals[self.next_subgoal_idx].literals 
            positive_lits = [l for l in lits if not l.is_negative]
            negative_lits_that_are_negated = [l.positive for l in lits if l.is_negative]
            if any(lit in next_state.literals for lit in negative_lits_that_are_negated):
                # There is no assignment that holds, since a negated lit in the subgoal is positive in the state.
                return
            if all(l in next_state.literals for l in positive_lits):
                # Check that all object names in the state literals match the object names in the goal
                self.precondition_targeting = True
                logging.info(f"ACHIEVED SUBGOAL {self.subgoals[self.next_subgoal_idx]}")
                self.next_subgoal_idx += 1
                self.action_seq.extend(self.actions_since_last_subgoal)
                self.actions_since_last_subgoal = []
                self._plan_to_next_subgoal = None
                self._last_plan_to_next_subgoal = None

        
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
        some_learned_operator_changed, updated_action_pred_names = self._operator_learning_module.learn(itr)
        logging.info(f"Learning took {time.time() - start} s")

        # Used in LLMIterative only
        if self.operator_learning_name in ['LLM+LNDR', 'LLMIterative+LNDR']:
            self._curiosity_module.learn(itr)

        if some_learned_operator_changed:
            self._curiosity_module.learning_callback()
            # only remove the operators for the action predicates that have been updated
            removes = set()
            for op_name in self._ops_preconds_executed:
                if op_name.rstrip('0123456789') in updated_action_pred_names:
                    removes.add(op_name)
            for r in removes:
                self._ops_preconds_executed.remove(r)

            # replan to the next subgoal.
            self._plan_to_next_subgoal = None

        else:
            # If the operators don't change as a result of adding the NOP, then the op preconds should be added to the visited set.
            if self._plan_to_op_preconds_failed:
                # This may be None if the plan to the operator preconds was length 1, so the visited set already contains the operator preconditions.
                if self._op_preconds_to_execute is not None:
                    self._ops_preconds_executed.add(self._op_preconds_to_execute)
                    self._op_preconds_to_execute = None
        return some_learned_operator_changed, some_learned_operator_changed
    
class InteractiveAgentLifted(InteractiveAgentGrounded):
    def __init__(self, domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path:Optional[str]):
        super().__init__(domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path)
        self._curiosity_module._ignore_mutex = False
        self._curiosity_module._ignore_statics = False
        self._curiosity_module._compute_goals = False
 
    def _get_action_with_preconds_as_goals(self, state, ops_to_exclude):
        # Have successfully executed the plan to the operator preconds, and will execute the operator next
        if self._preconds_plan is not None and len(self._preconds_plan) == 0:
            self.finished_preconds_plan = True
            self.actions_since_last_subgoal = []
            self._ops_preconds_executed.add(self._op_preconds_to_execute)
            self._op_preconds_to_execute = None
            self._preconds_plan = None
            # Ground action
            goal, lifted_act = self._current_goal_action
            ground_act = self._curiosity_module._sample_action_from_goal(goal, lifted_act,state, self._rand_state)
            logging.info(f"GROUNDED ACTION: {ground_act}")
            return ground_act

        # Follow plan to the operator's preconditions
        elif self._preconds_plan is not None and len(self._preconds_plan) > 0:
            self.finished_preconds_plan = False
            logging.info(f"FOLLOWING PLAN: {self._preconds_plan}")
            return self._preconds_plan.pop(0)

        action_predicates = set(p.name for p in self.action_space.predicates)
        for op in self._rand_state.permutation(sorted(self.learned_operators, key=lambda op: op.name)):
            # since the last time operators were learned, if operator has been successfully executed at the end of the plan, or
            # the plan failed in the middle to the operator preconditions, skip it.
            if op.name in self._ops_preconds_executed:
                continue
            if op.name in ops_to_exclude: continue
            preconds = op.preconds.literals

            logging.info(f"Trying preconds for op: {op.name}: {preconds}")

            # plan to lifted preconditions.
            preconds_hash = get_hashable_preconds_action(preconds)
            lifted_act = [p for p in preconds if p.predicate.name in action_predicates][0]
            if (preconds_hash, state) in self._visited_preconds_states[lifted_act.predicate]:
                continue
            lifted_precond_no_act = [p for p in preconds if p.predicate.name not in action_predicates]
            variables = sorted({ v for lit in lifted_precond_no_act for v in lit.variables })
            logging.info(f"variables: {variables}")
            # add differents
            if self.domain_name == 'Bakingrealistic':
                Different = Predicate('different', 2)
                for param1 in variables:
                    param1_type = param1._str[param1._str.find(':'):]
                    for param2 in variables:
                        if param1._str >= param2._str:
                            continue
                        param2_type = param2._str[param2._str.find(':'):]

                        if param1_type == param2_type:
                            lifted_precond_no_act.append(Different(param1, param2))

            plan = self._get_plan_to_preconds(lifted_precond_no_act, state)
            self._current_goal_action = (tuple(lifted_precond_no_act), lifted_act)
            self._visited_preconds_states[lifted_act.predicate].add((preconds_hash, state))
            if plan == 'skip':
                self._ops_preconds_executed.add(op.name)
                break
            elif plan is not None:
                self._preconds_plan = plan
                logging.info(f"Found plan to preconds: {preconds_hash}")
                logging.info(f"PLAN: {self._preconds_plan}")
                self._op_preconds_to_execute = op.name
                if len(self._preconds_plan) == 0:
                    # ground action
                    goal, lifted_act = self._current_goal_action
                    ground_act = self._curiosity_module._sample_action_from_goal(goal, lifted_act,state, self._rand_state)
                    logging.info(f"GROUNDED ACTION: {ground_act}")
 
                    self.finished_preconds_plan = True
                    self.actions_since_last_subgoal = []
                    self._ops_preconds_executed.add(self._op_preconds_to_execute)
                    self._op_preconds_to_execute = None
                    self._preconds_plan = None
                    return ground_act
                else:
                    return self._preconds_plan.pop(0)

        # once done, proceed to the next subgoal in the file.
        return None

    def _get_plan_to_preconds(self, lifted_precond_lits:list, state):
        """Returns None if no plan found, otherwise a list of action literals."""
        variables = sorted({ v for lit in lifted_precond_lits for v in lit.variables })
        body = LiteralConjunction(lifted_precond_lits)
        goal = Exists(variables, body)
        logging.info(f"Planning to goal: {goal}")

        # Create a pddl problem file with the goal and current state
        problem_fname = self._curiosity_module._create_problem_pddl(
            state, goal, prefix='glibl_preconds')

        # Get a plan
        try:
            plan, _ = self._planning_module.get_plan(
                problem_fname, use_cache=False, use_learned_ops=True)
            os.remove(problem_fname)
            return plan
        except NoPlanFoundException:
            logging.info(f"No plan found.")
        except PlannerTimeoutException:
            logging.info(f"PLANNER TIMED OUT")
            # if input("skip this preconditions? y or anything").strip() == 'y':
            #     os.remove(problem_fname)
            #     return 'skip'

        os.remove(problem_fname)

        return None
 

class DemonstrationsAgent(Agent):
     def __init__(self, domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path:Optional[str]):
        super().__init__(domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path)
        self.name = 'demoagent'   

        # Load the demos
        demos_path = f'/home/ubuntu/GLIB-Baking-Fails-and-LLMs/demonstrations/{self.domain_name.lower()}_demonstrations.pkl'
        # demos_path = f'/home/catalan/GLIB-Baking-Fails-and-LLMs/demonstrations/{self.domain_name.lower()}_demonstrations.pkl'
        with open(demos_path, 'rb') as f:
            transitions = pickle.load(f)
        self._operator_learning_module._transitions = transitions
 
        for action_pred in transitions:
            self._operator_learning_module._fits_all_data[action_pred] = False
 
    
class CreateDemonstrationsAgent(Agent):
    """An agent with initial demonstration data to each of the 4 train tasks."""
    def __init__(self, domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path:Optional[str]):
        super().__init__(domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path)
        
        self.name = 'demos'
        # dict: problem index -> step in the plan to execute next
        self.problem_to_plan_step = {i: 0 for i in range(len(ac.train_env.problems))}

        # dict: problem index -> list of plan steps (ground action predicate strings)
        self.plans = {}
        self._get_plans()
        self.prev_episode_idx = None
        # Keep track of episodes that have finished at least once
        self.terminated_episodes = set()
        self.action_space = action_space
        self.finished_preconds_plan = False
        # for action_pred in transitions:
        #     self._operator_learning_module._fits_all_data[action_pred] = True
 
        

    def get_action(self, state, problem_idx, _precond_targeting_only):

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
        demos_path = f'/home/catalan/GLIB-Baking-Fails-and-LLMs/demonstrations/{self.domain_name.capitalize()}'
        for i, file in enumerate(sorted(os.listdir(demos_path))):
            filepath = os.path.join(demos_path, file)
            problem_i = int(file[len('problem'):-len('.txt')])
            with open(filepath, 'r') as f:
                self.plans[problem_i] = [l for l in f.readlines() if l.strip() != '']
    
    def reset_episode(self, state, subgoals_path):
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
    
class StudentAgent(InteractiveAgentLifted):

    # Number of lits to change in the goal.
    MAX_LIT_CHANGES = 3

    def __init__(self, domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path:Optional[str]):
        super().__init__(domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path)
 
        self.name = 'student'
        self._ops_executed = set()
        self._mode = "teacher_subgoals"
        self.plan = None
        self._ground_truth_operators = {deepcopy(op) for op in ac.train_env.domain.operators.values()}
        if self.domain_name == 'Bakingrealistic':
            domain_parser = PDDLDomainParser('/home/catalan/pddlgym/pddlgym/pddl/bakingrealistic.pddl')
            self._ground_truth_operators_for_planning = {deepcopy(domain_parser.operators[o]) for o in domain_parser.operators}
        else:
            self._ground_truth_operators_for_planning = self._ground_truth_operators
        obj_types = set()
        for p in (self.action_space.predicates + self.obs_space.predicates):
            for t in p.var_types:
                obj_types.add(t)
        self.parser = GoalParser({p.name: p for p in self.action_space.predicates}, {p.name: p for p in self.obs_space.predicates}, obj_types)
        self.operator_parser = LLM_PDDL_Parser({p.name: p for p in self.action_space.predicates}, {p.name: p for p in self.obs_space.predicates}, obj_types)
        self._action_in_plan_to_preconds = False       
        self._visited_preconds_states_teacher_mode = set()
        self._evaluated_before_exception = False
        self.finished_preconds_plan = False

        if self.domain_name == 'Bakingrealistic':
            for op in self._ground_truth_operators:
                all_pass = False
                while not all_pass:
                
                    all_pass = True
                    for i in range(len(op.preconds.literals)):
                        lit = op.preconds.literals[i]
                        if lit.predicate.name in ('different' , 'name-less-than'):
                            op.preconds.literals.pop(i)
                            all_pass = False
                            break
                params = set()
                for lit in op.preconds.literals + op.effects.literals:
                    for v in lit.variables:
                        params.add(v)
                op.params = sorted(params, key=lambda param: param._str.split(':')[0])
            for op in self._ground_truth_operators_for_planning:
                all_pass = False
                while not all_pass:
                
                    all_pass = True
                    for i in range(len(op.preconds.literals)):
                        lit = op.preconds.literals[i]
                        if lit.predicate.name == 'name-less-than':
                            op.preconds.literals.pop(i)
                            all_pass = False
                            break
                params = set()
                for lit in op.preconds.literals + op.effects.literals:
                    for v in lit.variables:
                        params.add(v)
                op.params = sorted(params, key=lambda param: param._str.split(':')[0])
            
            # Read transitions file.
        # with open('/home/catalan/GLIB-Baking-Fails-and-LLMs/transitions.pkl', 'rb') as f:
        #     self._operator_learning_module._transitions = pickle.load(f)
        # with open('/home/catalan/GLIB-Baking-Fails-and-LLMs/ndrs.pkl', 'rb') as f:
        #     self._operator_learning_module._ndrs = pickle.load(f)

        # for action_pred in self._operator_learning_module._transitions:
        #     self._operator_learning_module._fits_all_data[action_pred] = False
                

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

        # Check if planned to preconditions
        if self._action_in_plan_to_preconds:
            # Stop executing the plan if it failed in the middle.
            if len(effects) == 0:
                self.finished_preconds_plan = True
                # About to reset to the previous subgoal, so clear this list.
                self.actions_since_last_subgoal = []
                self._preconds_plan = None
                # If the operators don't change as a result of adding the NOP, then the op preconds should be added to the visited set.
                self._plan_to_op_preconds_failed = True
            else:
                self._plan_to_op_preconds_failed = False
                
        else:
            if len(effects) == 0:
                logging.info(f"Setting plan to none ")
                self.plan = None
                # set flag to reset to start state.
                return True


        # Check if planned to the next subgoal
 
    def get_action(self, state,  _problem_idx, precond_targeting_only):

        if self.plan is not None:
            return self._execute_plan(self.plan, state)

        self._action_in_plan_to_preconds = False
        operator_names_tried = set()
        all_operator_names = {o.name for o in self.learned_operators}
        while operator_names_tried != all_operator_names:

            self._skip_to_next_op = False
            ### First step: operator matching

            OP = None
            for o in np.random.permutation(sorted(self.learned_operators, key = lambda operator: operator.name)):
                if o.name not in operator_names_tried:
                    OP = o
                    logging.info(f"Selected op: {OP.pddl_str()}")
                    break
            action_pred = [l.predicate for l in OP.preconds.literals if l.predicate in self.action_space.predicates][0]


            # Group ops by action predicate.
            ops_covered = [OP.name]
            ops_to_consider = []
            for o in self.learned_operators:
                if o.name == OP.name: continue
                a = [l.predicate for l in o.preconds.literals if l.predicate in self.action_space.predicates][0]           
                if a == action_pred:
                    ops_to_consider.append(o)
            
            # Ask to join as many operators as possible.
            while True:
                    
                logging.info(f"Ops to consider: {ops_to_consider}")
                try:
                    if not (len(ops_to_consider) == 0 or 'use-stand-mixer' in OP.name):
                        file = input("File containing merged operator or q? ").strip()

                        if file == 'd':
                            dump_intermediate_state(self)
                            logging.info("Dumped state")

                        while file not in ('q', 'qq') and not os.path.exists(file):
                            file = input("File containing merged operator or q? ").strip()

                            if file == 'qq':
                                # skip this operator
                                self._skip_to_next_op = True
                                break
                            elif file == 'd':
                                dump_intermediate_state(self)
                                logging.info("Dumped state")
                            elif file == 'q':
                                pass
                            else:
                                with open(file, 'r') as f:
                                    operator_str = ''.join(f.readlines())
                                
                                OP = self.operator_parser.parse_operators(operator_str)[0]
                                logging.info(f"parsed user joined operator: {OP.pddl_str()}")
                                break
                        if self._skip_to_next_op:
                            break

                    logging.info(f"Looking for g.t. operator that matches operator.")
                    # Compare the joined learned operator effects to the ground truth operators effects.
                    ground_truth_operator = None
                    for op in self._ground_truth_operators:
                        # logging.info(f"Checking if equal: {op.name}")
                        if effects_equal(op, OP):
                            ground_truth_operator = op
                            break
                    if ground_truth_operator is None:
                        name = input("g.t. operator name or 'q' to manually enter goal").strip()
                        names = {o.name for o in self._ground_truth_operators}
                        while name not in names and name != 'q':
                            name = input("g.t. operator name or 'q' to manually enter goal").strip()                           
                        if name == 'q':
                            plan = self._prompt_for_grounded_goal_and_plan(state, OP)
                            if plan not in ('q', 'qq'):
                                return plan
                            elif plan == 'qq':
                                break
                        ground_truth_operator = [o for o in self._ground_truth_operators if o.name == name][0]

                    if self._skip_to_next_op:
                        break
                    assert ground_truth_operator is not None, "No G.T. operator found. Unexpected."
                    logging.info(f"Matched with ground truth operator: {ground_truth_operator.pddl_str()}")
                    break
                except Exception as e:
                    print(e)
                    traceback.print_exc() 
                    input("Continue or Ctrl-C to quit:")
                    continue

            if self._skip_to_next_op:
                continue
            
            ### Second step: goal selection.

            preconds_changes = {'weak': [], 'strong': []}
            relation = None
            intersection_preconds = []

            # rename the params in g.t. op starting from ?x0: create a copy of this operator.
            ground_truth_operator = deepcopy(ground_truth_operator)
            param_mapping = {}
            max_effects_param_i = -1
            param_i = 0
            for lit in ground_truth_operator.effects.literals:
                for v in lit.variables:
                    if v not in param_mapping:
                        param_mapping[v] = TypedEntity(f'?x{param_i}', Type(v._str.split(':')[1]))
                        max_effects_param_i = max(param_i, max_effects_param_i)
                        param_i += 1
            for lit in ground_truth_operator.preconds.literals:
                for v in lit.variables:
                    if v not in param_mapping:
                        param_mapping[v] = TypedEntity(f'?x{param_i}', Type(v._str.split(':')[1]))
                        param_i += 1
            for lit in ground_truth_operator.preconds.literals:
                lit.set_variables([param_mapping[v] for v in lit.variables])
            for lit in ground_truth_operator.effects.literals:
                lit.set_variables([param_mapping[v] for v in lit.variables])
            ground_truth_operator.params = set(param_mapping.values())
                    
            # match the params in the learned op using the effects and search over the remaining parameters in the preconditions to maximize the number of lits that match in the preconds
            learned_operator = deepcopy(OP)
            learned_operator = reparameterize_learned_operator_by_matching_effects(learned_operator, ground_truth_operator)

            # given that parameterization, identify the weak/strong literals
                
            # Get the intersection and strong lits
            lit_i = 0
            for lit in learned_operator.preconds.literals:
                if lit in ground_truth_operator.preconds.literals:
                    intersection_preconds.append(lit)
                else:
                    preconds_changes['strong'].append((f'strong{lit_i}', lit)) 
                    relation = 'strong'
                lit_i += 1

            lit_i = 0

            for lit in ground_truth_operator.preconds.literals:
                if lit not in learned_operator.preconds.literals:
                    if relation == 'strong':
                        relation = 'mixed'
                    elif relation is None:
                        relation = 'weak'

                    preconds_changes['weak'].append((f'weak{lit_i}', lit))
                    lit_i += 1
            
            logging.info(f'gt. operator: {ground_truth_operator.pddl_str()}')
            logging.info(f'learned operator: {learned_operator.pddl_str()}')

            banks = []
            strong_base_preconds = deepcopy(ground_truth_operator.preconds.literals)
            # Add the action predicate
            if len([lit for lit in strong_base_preconds if lit.predicate in self.action_space.predicates]) == 0:
                action_pred = [act_pred for act_pred in self.action_space.predicates if act_pred.name == learned_operator.name.rstrip('0123456789')][0]
                strong_base_preconds.append(action_pred(*sorted(op.params, key=lambda param: param._str.split(':')[0])))

            if relation == 'weak':
                base_preconds = deepcopy(learned_operator.preconds.literals)
                banks.append((base_preconds, preconds_changes['weak']))
            elif relation == 'strong':
                banks.append((strong_base_preconds, preconds_changes['strong']))
            else:
                banks.append((deepcopy(learned_operator.preconds.literals) ,preconds_changes['weak']))
                banks.append((strong_base_preconds,preconds_changes['strong']))

            for base_preconds, changes_bank in banks:
                logging.info(f'Change bank length: {len(changes_bank)}')
                logging.info(changes_bank)
                # if change bank is too long (> 4), then skip straight to requesting the grounded goal / dump transitions / skip this operator.
                if len(changes_bank) > 4:
                    plan = self._prompt_for_grounded_goal_and_plan(state, learned_operator)
                    if plan not in ('q', 'qq'):
                        return plan
                    elif plan == 'qq':
                        break

                for n in range(1, min(len(changes_bank), self.MAX_LIT_CHANGES) + 1)[::-1]:
                    for changes in itertools.combinations(changes_bank, n):
                        goal = [l for l in base_preconds]
                        for change_type, lit in changes:
                            # change the lit in the goal
                            for i in range(len(goal)):
                                goal_lit = goal[i]
                                if goal_lit.negative == lit or goal_lit.positive == lit:
                                    goal.pop(i)
                                    break

                            if lit.is_negative:
                                goal.append(lit.positive)
                            else:
                                goal.append(lit.negative)

                        # only add to visited if the plan completes
                        mark = get_hashable_preconds_action(tuple(sorted(goal)))
                        if (mark, learned_operator.pddl_str()) in self._visited_preconds_states_teacher_mode:
                            logging.info(f"Skipping goal: {goal}")
                            continue

                        goal_no_action = [l for l in goal if goal if l.predicate not in self.action_space.predicates]
                        vars_ = sorted({ v for lit in goal_no_action for v in lit.variables })
                        # add differents
                        if self.domain_name == 'Bakingrealistic':
                            Different = Predicate('different', 2)
                            for param1 in vars_:
                                param1_type = param1._str[param1._str.find(':'):]
                                for param2 in vars_:
                                    if param1._str >= param2._str:
                                        continue
                                    param2_type = param2._str[param2._str.find(':'):]

                                    if param1_type == param2_type:
                                        goal_no_action.append(Different(param1, param2))


                        lifted_act = [l for l in base_preconds if l.predicate in self.action_space.predicates][0]
                        self._current_goal_action_operator = (goal_no_action, lifted_act, learned_operator.pddl_str())
                        body = LiteralConjunction(goal_no_action)
                        vars_ = sorted({ v for lit in body.literals for v in lit.variables })
                        goal = Exists(vars_, body)
                        logging.info(f"SAMPLED GOAL: {goal}\nACTION: {lifted_act}")
                        plan =  self._get_ground_truth_plan(goal, state)
                        if plan not in (None, -1):
                            logging.info(f"FOUND PLAN UNDER GT OPS: {plan}")
                            self._evaluated_before_exception = False
                            return self._execute_plan(plan, state)
                        elif plan == -1:
                            # no plan found.
                            # mark this goal as visited
                            self._visited_preconds_states_teacher_mode.add((mark, learned_operator.pddl_str()))
                            continue
                        else:
                            # planner timed out.
                            plan = self._prompt_for_grounded_goal_and_plan(state, learned_operator)
                            if plan not in ('q', 'qq'):
                                return plan
                        if self._skip_to_next_op:
                            break
                    if self._skip_to_next_op:
                        break
                if self._skip_to_next_op:
                    break

            operator_names_tried.update(ops_covered)
            ###

        option_str = \
"""Please pick an option:
*** Utils ***
[5] Dump the transitions and operators.
[9] Evaluate operators.
[11] End experiment.
[12] Restart cycle.
"""
        try:
            option = int(input(option_str).strip())
        except:
            option = None
        while option is None or (option not in [9,11,12]):
            try:
                if option == 5:
                    logging.info("Dumping state.")
                    dump_intermediate_state(self)
                option = int(input(option_str).strip())
            except:
                option = None           
        self.option = option
        return None
            
    def _prompt_for_grounded_goal_and_plan(self, state, operator):
        timeout = ac.planner_timeout 
        ac.planner_timeout = 400
        # provide the grounded goal file according to the lifted goal and then plan to it.
        while True:
            # if option is qq, then skip all goals for this operator.
            goal_file = input("Enter the grounded goal file: ").strip()
            if goal_file == 'qq':
                self._skip_to_next_op = True
                break                       
            elif goal_file == 'q':
                break
            elif goal_file == 'd':
                dump_intermediate_state(self)
                logging.info("Dumped state")
            try:
                with open(goal_file, 'r') as f:
                    lines = f.readlines()
                goal_lits = []
                for literal_str in lines[0].split(','):
                    literal_str = literal_str.strip()[1:-1]
                    if literal_str.startswith('not '):
                        literal_str = literal_str[len('not '):]
                        literal_str = literal_str[1:-1]
                        items = literal_str.split()
                        pred = Not(self._get_predicate(items[0], items[1:], state.objects))
                    else:
                        items = literal_str.split()
                        pred = self._get_predicate(items[0], items[1:], state.objects)
                    goal_lits.append(pred)
                logging.info(f"parsed user goal: {goal_lits}")
                ground_act = [lit for lit in goal_lits if lit.predicate in self.action_space.predicates][0]
                g = [lit for lit in goal_lits if lit.predicate not in self.action_space.predicates]
                goal = LiteralConjunction(g)
                self._current_goal_action_operator = (g, ground_act, operator.pddl_str())
                plan = self._get_ground_truth_plan(goal, state)
                assert plan not in (None, -1)
                logging.info(f"FOUND PLAN: {plan}")
                ac.planner_timeout = timeout
                return self._execute_plan(plan, state)
            except Exception as e:
                print(e)
                traceback.print_exc() 
                input("Continue or Ctrl-C to quit:")
                continue
        ac.planner_timeout = timeout
        if goal_file == 'qq' or goal_file == 'q':
            return goal_file

    def _get_predicate(self, pred_name:str, object_names:list, objects:frozenset):
        pred = [p for p in self.obs_space.predicates + self.action_space.predicates if p.name == pred_name][0]
        args = []
        for object_name in object_names:
            for o in objects:
                obj_name, _ = o._str.split(':')
                if obj_name == object_name:
                    args.append(o)
                    break
        return pred(*args)
 
    def _get_plan(self, goal, state):
        # Create a pddl problem file with the goal and current state
        problem_fname = self._curiosity_module._create_problem_pddl(
            state, goal, prefix='glibl_preconds')

        # Get a plan
        try:
            plan, _ = self._planning_module.get_plan(
                problem_fname, use_cache=False, use_learned_ops=True)
            os.remove(problem_fname)
            return plan
        except NoPlanFoundException:
            logging.info(f"No plan found.")
            os.remove(problem_fname)
            return -1
        except PlannerTimeoutException:
            logging.info(f"PLANNER TIMED OUT")
            os.remove(problem_fname)
            return None
 
    
    def _get_ground_truth_plan(self, goal, state):
        problem_fname = self._curiosity_module._create_problem_pddl(
            state, goal, prefix='glibl_preconds')
        # Get a plan
        try:
            if self.domain_name == 'Bakingrealistic':
                plan, _ = self._planning_module.get_plan(
                    problem_fname, use_cache=False, use_learned_ops=False, bakinglarge_file=True, ops=self._ground_truth_operators_for_planning)
                os.remove(problem_fname)
                return plan
 
            else:
                plan, _ = self._planning_module.get_plan(
                    problem_fname, use_cache=False, use_learned_ops=False, ops=self._ground_truth_operators_for_planning)
                os.remove(problem_fname)
                return plan
        except NoPlanFoundException:
            logging.info(f"No plan found.")

            os.remove(problem_fname)

            return -1
        except PlannerTimeoutException:
            logging.info(f"PLANNER TIMED OUT")

            os.remove(problem_fname)

            return None


    def _execute_plan(self, plan, state):

        self.plan = plan

        if len(self.plan) == 0:
            goal, act, operator_str = self._current_goal_action_operator
            if self._is_lifted(act):
                ground_act = self._curiosity_module._sample_action_from_goal(goal, act,state, self._rand_state)
            else:
                ground_act = act

            mark = get_hashable_preconds_action(tuple(sorted(goal)))
            self._visited_preconds_states_teacher_mode.add((mark, operator_str))
            self.plan = None
            self.finished_preconds_plan = True
            logging.info(f"Executing grounded action: {ground_act}")
            return ground_act

        return self.plan.pop(0)
    
    def _is_lifted(self, action):
        for v in action.variables:
            if '?' in v._str:
                return True

class StudentAgentSubgoals(StudentAgent):
    """
    An agent similar to StudentAgent, but instead of supplying a single grounded
    goal file, we supply a "goal-action file" with multiple subgoals (one per line).
    The last line contains a final grounded action to execute after achieving
    all subgoals in order.

    Behavior:
        1. On planner timeout or user request, load the subgoals + action from a file.
        2. For each subgoal (in order), attempt to plan and execute the plan.
           - If the plan fails mid-execution, reset plan variables and do any
             needed logic from StudentAgent.
        3. After all subgoals are achieved, execute the final grounded action
           on the last line of the file.
        4. If that fails, similarly reset as above.

    This class overrides get_action() to implement the above logic.
    """
    def __init__(self, domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path: Optional[str]):
        super().__init__(domain_name, action_space, observation_space,
                         curiosity_module_name, operator_learning_name,
                         planning_module_name, log_llm_path)
        self.name = 'student_subgoals'

        # List of subgoals to achieve in order, each subgoal is a LiteralConjunction
        self.subgoals = []
        # Which subgoal index we are currently trying to achieve, or -np.inf as the null index
        self.next_subgoal_idx = -np.inf
        # Plan to the current subgoal
        self.plan_to_next_subgoal = None
        # Executed actions since the last subgoal was achieved
        self.actions_since_last_subgoal = []
        # The final action to execute once all subgoals are done
        self._final_action = None
    
    def reset_episode(self, state, _):
        """Reset the episode and load subgoals from subgoals_file if provided."""
        # First do the StudentAgent's reset logic (which calls parent's reset too)
        super().reset_episode(state, _)

        # Clear our subgoal info
        self.subgoals = []
        self.next_subgoal_idx = -np.inf
        self.plan_to_next_subgoal = None
        self.actions_since_last_subgoal = []
        self._final_action = None


    def _load_subgoals(self, state, subgoals_file):
        """
        Read each line from the file, parse it into a list of grounded Literals.
        - All but the last line are subgoals (to be treated as conjunctive goals).
        - The last line is the final grounded action to execute after all subgoals.
        """
        with open(subgoals_file, "r") as f:
            lines = [ln.strip() for ln in f.readlines()]
            lines = [ln for ln in lines if ln]  # remove empty lines

        # If there's only 1 line, that would mean no subgoals, only a final action.
        if len(lines) == 0:
            logging.info("No subgoals loaded: file is empty!")
            return

        # The final line is a single action (e.g., "(some_action objA objB)")
        # All preceding lines are subgoals (one subgoal per line).
        # Each subgoal line might be something like: 
        #   "(on objA objB), (clean objB), (not (broken objB))"
        # We parse each line into a list of pddlgym.Literal or Not(...) objects.
        last_line = lines[-1]
        subgoal_lines = lines[:-1]

        # Parse each subgoal line
        for subg_line in subgoal_lines:
            if not subg_line:
                continue
            # e.g. subg_line = "(lit1 obj1 obj2), (not (lit2 obj1 obj3))"
            subgoal_lits = []
            for literal_str in subg_line.split(","):
                literal_str = literal_str.strip()
                # remove parentheses, handle "not(...)"
                # This is basically the same logic as in InteractiveAgentGrounded._load_subgoals
                if literal_str.startswith("(") and literal_str.endswith(")"):
                    literal_str = literal_str[1:-1]
                if literal_str.startswith("not "):
                    # "not (stuff ...)"
                    # remove 'not ', remove leading/trailing parentheses
                    inside = literal_str[len("not "):].strip()
                    if inside.startswith("(") and inside.endswith(")"):
                        inside = inside[1:-1].strip()
                    items = inside.split()
                    pred = Not(self._get_obs_predicate(items[0], items[1:], state.objects))
                else:
                    # parse the positive literal
                    items = literal_str.split()
                    pred = self._get_obs_predicate(items[0], items[1:], state.objects)
                subgoal_lits.append(pred)
            # Now combine them in a LiteralConjunction
            conj = LiteralConjunction(subgoal_lits)
            self.subgoals.append(conj)

        # Parse the final line as a single grounded action
        # e.g. (some_grounded_action objA objB)
        # We'll store it as a pddlgym Literal so that we can just execute it later.
        final_action_str = last_line
        if final_action_str.startswith("(") and final_action_str.endswith(")"):
            final_action_str = final_action_str[1:-1]
        items = final_action_str.split()
        pred_name = items[0]
        objects_ = items[1:]
        # find the matching action predicate
        act_pred = [p for p in self.action_space.predicates if p.name == pred_name]
        if len(act_pred) == 0:
            raise ValueError(f"Could not find action predicate {pred_name} in action_space.")
        act_pred = act_pred[0]

        # Convert each object name to the actual typed object
        typed_objs = []
        for obj_name in objects_:
            matched_obj = None
            for o in state.objects:
                # e.g. "sugar:Ingredient"
                o_str, _ = o._str.split(":")
                if o_str == obj_name:
                    matched_obj = o
                    break
            if matched_obj is None:
                raise ValueError(f"Could not find object {obj_name} in the state!")
            typed_objs.append(matched_obj)
        self._final_action = act_pred(*typed_objs)

        self.next_subgoal_idx = 0
        logging.info(f"Loaded subgoals from {subgoals_file}: {self.subgoals}")
        logging.info(f"Final action: {self._final_action}")

    def get_action(self, state, _problem_idx, precond_targeting_only: bool):
        """
        Overridden get_action:
          2. If we've achieved all subgoals, attempt the final action.
             - If it fails, handle it similarly to plan-fail logic.
          3. Otherwise, plan to the next subgoal (like InteractiveAgentGrounded).
             - If the plan is not None, pop the next action.
             - If the plan fails or times out, reset plan and do whatever
               prompting you want (or just do nothing if you're fully automated).
        """
        # 1) If we already finished all subgoals, attempt final action
        if self.next_subgoal_idx >= len(self.subgoals) and self._final_action is not None:

            self.finished_preconds_plan = True
            self.plan_to_next_subgoal = None
            self.next_subgoal_idx = None

            logging.info("All subgoals achieved. Executing final action.")
            # We don't plan for a single action; we just do it directly
            # (assuming the final action is guaranteed to be grounded).
            action = self._final_action
            self._final_action = None
            return action

        # 3) Otherwise, we are working on the next subgoal
        # If we already have a plan in progress, continue it
        if self.plan_to_next_subgoal is not None and len(self.plan_to_next_subgoal) > 0:
            # Pop the next action
            action = self.plan_to_next_subgoal.pop(0)
            return action
        elif  self.next_subgoal_idx != -np.inf and self.next_subgoal_idx < len(self.subgoals):
            # Attempt to plan to the next subgoal
            subgoal = self.subgoals[self.next_subgoal_idx]
            logging.info(f"Planning to subgoal {self.next_subgoal_idx}: {subgoal}")

            problem_fname = self._curiosity_module._create_problem_pddl(
                state, subgoal, prefix='glib_subgoal'
            )
            plan = None
            try:
                plan, _ = self._planning_module.get_plan(
                    problem_fname,
                    use_cache=False,
                    use_learned_ops=True
                )
            except NoPlanFoundException:
                logging.info("No plan found to subgoal.")
            except PlannerTimeoutException:
                logging.info("Planner timed out for subgoal.")
            finally:
                if os.path.exists(problem_fname):
                    os.remove(problem_fname)

            if plan:
                logging.info(f"Found plan to subgoal {self.next_subgoal_idx}: {plan}")
                self.plan_to_next_subgoal = plan

                self._current_goal_action_operator = (tuple(self.subgoals), self._final_action, chosen_op.pddl_str())

                # Execute the first step
                action = self.plan_to_next_subgoal.pop(0)
                self.actions_since_last_subgoal.append(action)
                return action
            else:
                # Plan not found or timed out, you can do manual logic here
                # or just return None. Possibly reset plan, do user prompting, etc.
                logging.info("Plan to subgoal failed/timed out. Resetting plan.")
                self.plan_to_next_subgoal = None
                return None
        else:
            # 4) choose an operator and try informative goals: if planner times out or too many informative goals in the change bank, then prompt user for subgoals list, like in StudentAgent.
            logging.info("=== Step 4) operator-based 'informative goals' approach (StudentAgent style) ===")

            # Keep track of which operator names have already been tried
            operator_names_tried = set()
            all_operator_names = {op.name for op in self.learned_operators}

            # We will attempt to create a small "change bank" of goals for each operator
            # by flipping or negating some preconditions, etc.
            # This logic is extremely domain / application dependent, but here's a template.
            while operator_names_tried != all_operator_names:

                self._skip_to_next_op = False
                # 4a) pick an untried operator
                chosen_op = None
                for op in self._rand_state.permutation(sorted(self.learned_operators, key=lambda o: o.name)):
                    if op.name not in operator_names_tried:
                        chosen_op = op
                        break
                action_pred = [l.predicate for l in chosen_op.preconds.literals if l.predicate in self.action_space.predicates][0]

                ops_to_consider = []
                for o in self.learned_operators:
                    if o.name == chosen_op.name: continue
                    a = [l.predicate for l in o.preconds.literals if l.predicate in self.action_space.predicates][0]           

                    if a == action_pred:
                        ops_to_consider.append(o)
    
                # Ask to join as many operators as possible.
                while True:
                        
                    logging.info(f"Ops to consider: {ops_to_consider}")
                    try:
                        if not (len(ops_to_consider) == 0 or 'use-stand-mixer' in chosen_op.name):
                            file = input("File containing merged operator or q? ").strip()

                            if file == 'd':
                                dump_intermediate_state(self)
                                logging.info("Dumped state")

                            while file not in ('q', 'qq') and not os.path.exists(file):
                                file = input("File containing merged operator or q? ").strip()

                                if file == 'qq':
                                    # skip this operator
                                    self._skip_to_next_op = True
                                    break
                                elif file == 'd':
                                    dump_intermediate_state(self)
                                    logging.info("Dumped state")
                                elif file == 'q':
                                    pass
                                else:
                                    with open(file, 'r') as f:
                                        operator_str = ''.join(f.readlines())
                                    
                                    chosen_op = self.operator_parser.parse_operators(operator_str)[0]
                                    logging.info(f"parsed user joined operator: {chosen_op.pddl_str()}")
                                    break
                            if self._skip_to_next_op:
                                break
                        logging.info(f"Looking for g.t. operator that matches operator:\n{chosen_op.pddl_str()}")
                        # Compare the joined learned operator effects to the ground truth operators effects.
                        ground_truth_operator = None
                        for op in self._ground_truth_operators:
                            if effects_equal(op, chosen_op):
                                ground_truth_operator = op
                                break
                        if ground_truth_operator is None:
                            name = input("g.t. operator name or 'q' to manually enter goal").strip()
                            names = {o.name for o in self._ground_truth_operators}
                            while name not in names and name != 'q':
                                name = input("g.t. operator name or 'q' to manually enter goal").strip()                           
                            if name == 'q':
                                plan = self._prompt_for_grounded_goal_and_plan(state, chosen_op)
                                if plan not in ('q', 'qq'):
                                    return plan
                                elif plan == 'qq':
                                    break
                            ground_truth_operator = [o for o in self._ground_truth_operators if o.name == name][0]

                        if self._skip_to_next_op:
                            break
                        assert ground_truth_operator is not None, "No G.T. operator found. Unexpected."
                        logging.info(f"Matched with ground truth operator: {ground_truth_operator.pddl_str()}")
                        break
                    except Exception as e:
                        print(e)
                        traceback.print_exc() 
                        input("Continue or Ctrl-C to quit:")
                        continue

                if self._skip_to_next_op:
                    continue
 

                logging.info(f"[Operator selection] Trying operator: {chosen_op.name}")

                preconds_changes = {'weak': [], 'strong': []}
                relation = None
                intersection_preconds = []

                # rename the params in g.t. op starting from ?x0: create a copy of this operator.
                ground_truth_operator = deepcopy(ground_truth_operator)
                param_mapping = {}
                max_effects_param_i = -1
                param_i = 0
                for lit in ground_truth_operator.effects.literals:
                    for v in lit.variables:
                        if v not in param_mapping:
                            param_mapping[v] = TypedEntity(f'?x{param_i}', Type(v._str.split(':')[1]))
                            max_effects_param_i = max(param_i, max_effects_param_i)
                            param_i += 1
                for lit in ground_truth_operator.preconds.literals:
                    for v in lit.variables:
                        if v not in param_mapping:
                            param_mapping[v] = TypedEntity(f'?x{param_i}', Type(v._str.split(':')[1]))
                            param_i += 1
                for lit in ground_truth_operator.preconds.literals:
                    lit.set_variables([param_mapping[v] for v in lit.variables])
                for lit in ground_truth_operator.effects.literals:
                    lit.set_variables([param_mapping[v] for v in lit.variables])
                ground_truth_operator.params = set(param_mapping.values())
                        
 
                # match the params in the learned op using the effects and search over the remaining parameters in the preconditions to maximize the number of lits that match in the preconds
                learned_operator = deepcopy(chosen_op)
                learned_operator = reparameterize_learned_operator_by_matching_effects(learned_operator, ground_truth_operator)

                # 4b) Build one or more "informative goals" from chosen_op's preconds
                # given that parameterization, identify the weak/strong literals
                    
                # Get the intersection and strong lits
                lit_i = 0
                for lit in learned_operator.preconds.literals:
                    if lit in ground_truth_operator.preconds.literals:
                        intersection_preconds.append(lit)
                    else:
                        preconds_changes['strong'].append((f'strong{lit_i}', lit)) 
                        relation = 'strong'
                    lit_i += 1

                lit_i = 0

                for lit in ground_truth_operator.preconds.literals:
                    if lit not in learned_operator.preconds.literals:
                        if relation == 'strong':
                            relation = 'mixed'
                        elif relation is None:
                            relation = 'weak'

                        preconds_changes['weak'].append((f'weak{lit_i}', lit))
                        lit_i += 1
                
                logging.info(f'gt. operator: {ground_truth_operator.pddl_str()}')
                logging.info(f'learned operator: {learned_operator.pddl_str()}')
                banks = []
                strong_base_preconds = deepcopy(ground_truth_operator.preconds.literals)
                # Add the action predicate
                if len([lit for lit in strong_base_preconds if lit.predicate in self.action_space.predicates]) == 0:
                    action_pred = [act_pred for act_pred in self.action_space.predicates if act_pred.name == learned_operator.name.rstrip('0123456789')][0]
                    strong_base_preconds.append(action_pred(*sorted(op.params, key=lambda param: param._str.split(':')[0])))

                if relation == 'weak':
                    base_preconds = deepcopy(learned_operator.preconds.literals)
                    banks.append((base_preconds, preconds_changes['weak']))
                elif relation == 'strong':
                    banks.append((strong_base_preconds, preconds_changes['strong']))
                else:
                    banks.append((deepcopy(learned_operator.preconds.literals) ,preconds_changes['weak']))
                    banks.append((strong_base_preconds,preconds_changes['strong']))

                for base_preconds, changes_bank in banks:
                    logging.info(f'Change bank length: {len(changes_bank)}')
                    logging.info(changes_bank)
                    # if change bank is too long (> 4), then skip straight to requesting the grounded goal / dump transitions / skip this operator.
                    if len(changes_bank) > 4:
                        plan = self._prompt_for_grounded_goal_and_plan(state, learned_operator)
                        if plan not in ('q', 'qq'):
                            return plan
                        elif plan == 'qq':
                            break

                    for n in range(1, min(len(changes_bank), self.MAX_LIT_CHANGES) + 1)[::-1]:
                        for changes in itertools.combinations(changes_bank, n):
                            goal = [l for l in base_preconds]
                            for change_type, lit in changes:
                                # change the lit in the goal
                                for i in range(len(goal)):
                                    goal_lit = goal[i]
                                    if goal_lit.negative == lit or goal_lit.positive == lit:
                                        goal.pop(i)
                                        break

                                if lit.is_negative:
                                    goal.append(lit.positive)
                                else:
                                    goal.append(lit.negative)

                            # only add to visited if the plan completes
                            mark = get_hashable_preconds_action(tuple(sorted(goal)))
                            if (mark, learned_operator.pddl_str()) in self._visited_preconds_states_teacher_mode:
                                logging.info(f"Skipping goal: {goal}")
                                continue

                            goal_no_action = [l for l in goal if goal if l.predicate not in self.action_space.predicates]
                            vars_ = sorted({ v for lit in goal_no_action for v in lit.variables })
                            # add differents
                            if self.domain_name == 'Bakingrealistic':
                                Different = Predicate('different', 2)
                                for param1 in vars_:
                                    param1_type = param1._str[param1._str.find(':'):]
                                    for param2 in vars_:
                                        if param1._str >= param2._str:
                                            continue
                                        param2_type = param2._str[param2._str.find(':'):]

                                        if param1_type == param2_type:
                                            goal_no_action.append(Different(param1, param2))


                            lifted_act = [l for l in base_preconds if l.predicate in self.action_space.predicates][0]
                            self._current_goal_action_operator = (goal_no_action, lifted_act, learned_operator.pddl_str())
                            body = LiteralConjunction(goal_no_action)
                            vars_ = sorted({ v for lit in body.literals for v in lit.variables })
                            goal = Exists(vars_, body)
                            logging.info(f"SAMPLED GOAL: {goal}\nACTION: {lifted_act}")
                            plan =  self._get_ground_truth_plan(goal, state)
                            if plan not in (None, -1):
                                logging.info(f"FOUND PLAN UNDER GT OPS: {plan}")
                                self._evaluated_before_exception = False
                                return self._execute_plan(plan, state)
                            elif plan == -1:
                                # no plan found.
                                # mark this goal as visited
                                self._visited_preconds_states_teacher_mode.add((mark, learned_operator.pddl_str()))
                                continue
                            else:
                                # planner timed out.
                                plan = self._prompt_for_grounded_goal_and_plan(state, learned_operator)
                                if plan not in ('q', 'qq'):
                                    return plan
                            if self._skip_to_next_op:
                                break
                        if self._skip_to_next_op:
                            break
                    if self._skip_to_next_op:
                        break
 
                # If we exhaust all candidate goals for chosen_op and none yield a plan,
                # we move on to the next operator. That’s the “while operator_names_tried != ...” loop.

                # Mark this operator as tried
                operator_names_tried.add(chosen_op.name)

            # 4d) If we exit this loop, it means we've tried all operators and failed 
            # to produce any plan. We can now do exactly what StudentAgent does: 
            # prompt for new subgoals, or simply return None, or something else.
            option_str = \
    """Please pick an option:
    *** Utils ***
    [5] Dump the transitions and operators.
    [9] Evaluate operators.
    [11] End experiment.
    [12] Restart cycle.
    """
            try:
                option = int(input(option_str).strip())
            except:
                option = None
            while option is None or (option not in [9,11,12]):
                try:
                    if option == 5:
                        logging.info("Dumping state.")
                        dump_intermediate_state(self)
                    option = int(input(option_str).strip())
                except:
                    option = None           
            self.option = option
            return None

    def _prompt_for_grounded_goal_and_plan(self, state, operator):
        timeout = ac.planner_timeout 
        ac.planner_timeout = 400
        # provide the grounded goal file according to the lifted goal and then plan to it.
        while True:
            # if option is qq, then skip all goals for this operator.
            goal_file = input("Enter the subgoal file: ").strip()
            if goal_file == 'qq':
                self._skip_to_next_op = True
                break                       
            elif goal_file == 'q':
                break
            elif goal_file == 'd':
                dump_intermediate_state(self)
                logging.info("Dumped state")
            try:
                self._load_subgoals(state, goal_file)
                plan = self._get_ground_truth_plan(self.subgoals[self.next_subgoal_idx], state)
                assert plan not in (None, -1)
                logging.info(f"FOUND PLAN: {plan}")
                while plan == []:
                    self.next_subgoal_idx += 1
                    if self.next_subgoal_idx > len(self.subgoals):
                        raise Exception("All subgoals are already achieved in the current state.")
                    plan = self._get_ground_truth_plan(self.subgoals[self.next_subgoal_idx], state)
                self.plan_to_next_subgoal = plan
                self._current_goal_action_operator = (tuple(self.subgoals), self._final_action, operator.pddl_str())
                ac.planner_timeout = timeout
                return self._execute_plan(plan, state)
            except Exception as e:
                print(e)
                traceback.print_exc() 
                input("Continue or Ctrl-C to quit:")
                continue
        ac.planner_timeout = timeout
        if goal_file == 'qq' or goal_file == 'q':
            return goal_file

    def _execute_plan(self, plan, state):
        assert not (len(self.plan_to_next_subgoal) == 0 and self.next_subgoal_idx < len(self.subgoals)), "unexpected"

        self.plan_to_next_subgoal = plan

        goals, act, operator_str = self._current_goal_action_operator
        if len(self.plan_to_next_subgoal) == 0 and self.next_subgoal_idx == len(self.subgoals):
            logging.info(f"Executing grounded action: {ground_act}")
            self.finished_preconds_plan = True
            self.plan_to_next_subgoal = None
 
            action =  self._final_action
            self._final_action = None
            return action

        elif len(self.plan_to_next_subgoal) == 0:
            # Sampled lifted goal
            ground_act = self._curiosity_module._sample_action_from_goal(goals, act,state, self._rand_state)
            mark = get_hashable_preconds_action(tuple(sorted(goals)))
            self._visited_preconds_states_teacher_mode.add((mark, operator_str))
            self.plan_to_next_subgoal = None
            self.finished_preconds_plan = True
            logging.info(f"Executing grounded action: {ground_act}")
            return ground_act

        return self.plan_to_next_subgoal.pop(0)
 
    def observe(self, state, action, next_state, itr):
        """
        Overridden observe to handle:
            - If the plan fails (no effects), reset plan variables.
            - If subgoal is achieved, increment subgoal index, etc.

        Otherwise, reuse the parent's logic from StudentAgent.
        
        - From ChatGPT. Looks correct.
        """
        # Check if the action had no effects => plan might be failing
        effects = self._compute_effects(state, next_state)
        if len(effects) == 0:
            logging.info("Plan execution failed mid-subgoal: resetting subgoal plan.")
            self.plan_to_next_subgoal = None

        # Check if we have achieved the current subgoal
        # (only if we are in the middle of planning to subgoals)
        if self.next_subgoal_idx < len(self.subgoals):
            subgoal = self.subgoals[self.next_subgoal_idx]
            # For negative-literal subgoals, check if the positive version
            # is in next_state. If so, it's not satisfied.
            all_satisfied = True
            for lit in subgoal.literals:
                if lit.is_negative:
                    # e.g. lit = Not(...)
                    if lit.positive in next_state.literals:
                        all_satisfied = False
                        break
                else:
                    if lit not in next_state.literals:
                        all_satisfied = False
                        break

            if all_satisfied:
                logging.info(f"Subgoal {self.next_subgoal_idx} achieved!")
                # Move on
                self.plan_to_next_subgoal = None
                self.next_subgoal_idx += 1

        # After executed the action and achieved all subgoals
        elif self.next_subgoal_idx == len(self.subgoals) and self._final_action is None:
            self.subgoals = []
            self.next_subgoal_idx = -np.inf
            return True

def get_hashable_preconds_action(preconds):
    # Sort preconditions by alphabetical order of its string representation.
    strings = []
    for pre in preconds:
        if pre.is_negative:
            pred = f'NOT-{pre.predicate.name}'
        else:
            pred = pre.predicate.name
        strings.append(f'({pred}' + ','.join(pre.pddl_variables()) + ')')
    s = ','.join(sorted(strings))
    return s
    
def dump_intermediate_state(agent:InteractiveAgentGrounded, fname='transitions.pkl'):
    with open(fname, 'wb') as f:
        pickle.dump(agent._operator_learning_module._transitions, f)
    with open('ops.pkl', 'wb') as f:
        pickle.dump(agent.learned_operators, f)
    with open('visited_preconds.pkl', 'wb') as f:
        pickle.dump(agent._visited_preconds_states, f)
    with open('ops_visited.pkl', 'wb') as f:
        pickle.dump(agent._ops_preconds_executed, f)
    with open("ndrs.pkl", 'wb') as f:
        pickle.dump(agent._operator_learning_module._ndrs, f)
    with open('rand_state.pkl', 'wb') as f:
        rand_state = agent._rand_state.get_state()
        pickle.dump(rand_state, f)

    
def rename_variables_in_lits(given_lits:list, conditioned_mapping: dict = {}) -> Tuple[list, dict]:
    literals = deepcopy(sorted(given_lits))
    params = set()
    for lit in literals:
        for v in lit.variables:
            params.add(v) 
    variable_nums_taken = {int(v._str.split(':')[0][len("?x"):]) for v in conditioned_mapping.values()}
    i = 0
    rename_map = {}
    for v in params:
        if v in conditioned_mapping:
            rename_map[v] = conditioned_mapping[v]
        name, v_type = v._str.split(':')
        while i in variable_nums_taken:
            i += 1
        new_var = TypedEntity(f'?x{i}', Type(v_type))
        rename_map[v] = new_var 
        i += 1

    for lit in literals:
        lit.set_variables([rename_map[v] for v in lit.variables])

    return literals, rename_map

def effects_equal(op1, op2):
    """Returns True if the lifted effects of the operators are equal, False otherwise."""

    # renumber the variables in the effects from 0 for both operators.
    op1_preconds, op1_preconds_map = rename_variables_in_lits(op1.preconds.literals)
    op2_preconds, op2_preconds_map = rename_variables_in_lits(op2.preconds.literals)
    op1_effects, _ = rename_variables_in_lits(op1.effects.literals, op1_preconds_map)
    op2_effects, _ = rename_variables_in_lits(op2.effects.literals, op2_preconds_map)

    # # This IS A HEURISTIC that is incorrect for domains in general. covers the Antis in the effects that are already negative in the preconditions
    # for eff_lit in op1_effects:
    #     if eff_lit.is_anti and (eff_lit.inverted_anti not in op2_preconds) and (eff_lit not in op2_effects) and (eff_lit.inverted_anti not in op2_effects):
    #         op2_effects.append(eff_lit)
    # for eff_lit in op2_effects:
    #     if eff_lit.is_anti and (eff_lit.inverted_anti not in op1_preconds) and (eff_lit not in op1_effects) and (eff_lit.inverted_anti not in op1_effects):
    #         op1_effects.append(eff_lit)                       

    op1_effects = sorted(op1_effects)
    op2_effects = sorted(op2_effects)
    # logging.info(f"Comparing {op1_effects} to {op2_effects}")
    # Get all parameterizations of the op1 params.
        # get all the variable names in a list, and use itertools.permutations(var_names)
    op1_params_list = []
    for lit in op1_effects:
        for param in lit.variables:
            op1_params_list.append(param._str.split(':')[0])

    # If number of literals aren't equal, return False.
    predicate_name_counts_op1 = defaultdict(lambda: 0)
    predicate_name_counts_op2 = defaultdict(lambda: 0)
    # type_to_param_op1_effects = defaultdict(lambda: [])
    # type_to_param_op2_effects = defaultdict(lambda: [])
    for lit in op1_effects:
        p_name = f'{lit.predicate.name}-{lit.is_anti}-{lit.is_negative}'
        predicate_name_counts_op1[p_name] += 1
        # for v in lit.variables:
            # type_to_param_op1_effects[v._str.split(':')[1]].append(v)
    for lit in op2_effects:
        p_name = f'{lit.predicate.name}-{lit.is_anti}-{lit.is_negative}'
        predicate_name_counts_op2[p_name] += 1
        # for v in lit.variables:
            # type_to_param_op2_effects[v._str.split(':')[1]].append(v)
    
    if sorted(predicate_name_counts_op1.keys()) != sorted(predicate_name_counts_op2.keys()):
        return False

    for key in predicate_name_counts_op1:
        if predicate_name_counts_op1[key] != predicate_name_counts_op2[key]:
            # logging.info("Returned at 2")
            return False
    
    return True

def join_operators(op1, op2, new_op_name):
    """Returns a new operator based on op1's parameterization if these operators can be joined, or None if they can't be joined.

    FIXME resolved: We now search over actual parameterizations directly, rather than using stripped arguments.
    """
    # Find a reparameterization where the preconditions can be joined, or return fail if not found.

    # Reparameterize the variables in the operators starting from 0
    # op1_preconds, op1_preconds_mapping = rename_variables_in_lits(op1.preconds.literals)
    # op2_preconds, op2_preconds_mapping = rename_variables_in_lits(op2.preconds.literals)

    op1_preconds = deepcopy(op1.preconds.literals)
    op2_preconds = deepcopy(op2.preconds.literals)
    op2_preconds_mapping = {}

    # Gather all unique parameters from op1 preconditions
    op1_preconds_params_list = set()
    for lit in op1_preconds:
        for param in lit.variables:
            op1_preconds_params_list.add(param)
    op1_preconds_params_list = list(op1_preconds_params_list)

    # Try all permutations of the op1 preconditions' parameters
    for perm in itertools.permutations(op1_preconds_params_list):
        # Map the original variable list to the chosen permutation
        variables = dict(zip(op1_preconds_params_list, perm))

        # Apply this variable mapping to op1's preconditions
        preconds = []
        for l in op1_preconds:
            args = [variables[v] for v in l.variables]
            preconds.append(Literal(l.predicate, args))

        # Identify common preconditions that appear as a literal in one operator 
        # and its negation in the other, without stripping arguments.
        common_in_op1 = []
        common_in_op2 = []
        op2_preconds_set = set(op2_preconds)

        for lit in preconds:
            opposite_lit = lit.negative
            if opposite_lit in op2_preconds_set:
                common_in_op1.append(lit)
                common_in_op2.append(opposite_lit)
            elif lit in op2_preconds_set:
                common_in_op1.append(lit)               
                common_in_op2.append(lit)

        base_preconds = set(preconds) - set(common_in_op1)
        op2_preconds_remaining = set(op2_preconds) - set(common_in_op2)

        # Check if the remaining sets of preconditions are identical
        if base_preconds == op2_preconds_remaining:
            # Construct parameter mappings for effects
            op1_conditioned_mapping = {}
            op1_effects, op1_operator_mapping = rename_variables_in_lits(op1.effects.literals, op1_conditioned_mapping)
            op2_effects, op2_operator_mapping = rename_variables_in_lits(op2.effects.literals, op2_preconds_mapping)

            # Identify effect parameters not covered by the precondition mapping
            op1_effects_params_list = set() 
            for lit in op1_effects:
                for v in lit.variables:
                    if v not in op1_conditioned_mapping.values():
                        op1_effects_params_list.add(v)
            op1_effects_params_list = list(op1_effects_params_list)

            # Try permutations of the remaining effect parameters
            for effects_perm in itertools.permutations(op1_effects_params_list):
                op1_effects_var_mapping = dict(zip(op1_effects_params_list, effects_perm))

                new_effects = []
                for lit in op1_effects:
                    args = [(op1_effects_var_mapping[v] if v in op1_effects_var_mapping else v) 
                            for v in lit.variables]
                    new_effects.append(Literal(lit.predicate, args))

                op1_effects = deepcopy(new_effects)

                # Carry over conditions to effects if possible (domain-specific logic)
                for lit in base_preconds:
                    if lit.predicate not in ac.train_env.action_space.predicates:
                        if lit.negative not in op1_effects:
                            op1_effects.append(lit)
                        if lit.negative not in op2_effects:
                            op2_effects.append(lit)

                # Domain-specific handling of antis
                for eff_lit in op1_effects:
                    if eff_lit.is_anti and (eff_lit.inverted_anti not in base_preconds) and \
                       (eff_lit.predicate not in [l.predicate for l in op2_effects]):
                        op2_effects.append(eff_lit)
                for eff_lit in op2_effects:
                    if eff_lit.is_anti and (eff_lit.inverted_anti not in base_preconds) and \
                       (eff_lit.predicate not in [l.predicate for l in op1_effects]):
                        op1_effects.append(eff_lit)
                        new_effects.append(eff_lit)

                # Check if effects match
                op1_effects_set = set(op1_effects)
                op2_effects_set = set(op2_effects)
                logging.info(f"Comparing {sorted(op1_effects)} to {sorted(op2_effects)}")

                if op1_effects_set == op2_effects_set:
                    params = set()
                    for lits in [op1_effects, base_preconds]:
                        for lit in lits:
                            for v in lit.variables:
                                params.add(v)
                    possible_operator =  Operator(new_op_name, params, 
                                    LiteralConjunction(list(base_preconds)), 
                                    LiteralConjunction(new_effects))
                    return possible_operator

    return None

def reparameterize_learned_operator_by_matching_effects(learned_operator, gt_operator):
    predicate_name_counts_op1 = defaultdict(lambda: (0, []))
    predicate_name_counts_op2 = defaultdict(lambda: (0, []))
    for lit in learned_operator.effects.literals:
        p_name = f'{lit.predicate.name}-{lit.is_anti}-{lit.is_negative}'
        count, l = predicate_name_counts_op1[p_name]
        l.append(lit)
        predicate_name_counts_op1[p_name] = (count + 1, l)
    for lit in gt_operator.effects.literals:
        p_name = f'{lit.predicate.name}-{lit.is_anti}-{lit.is_negative}'
        count, l = predicate_name_counts_op2[p_name]
        l.append(lit)
        predicate_name_counts_op2[p_name] = (count +1, l)
 
    assert all(predicate_name_counts_op1[k][0] == predicate_name_counts_op2[k][0] for k in predicate_name_counts_op1)

    param_mapping = {}
    for lit in learned_operator.effects.literals:
        p_name = f'{lit.predicate.name}-{lit.is_anti}-{lit.is_negative}'
        
        if predicate_name_counts_op2[p_name][0] == 1:
            for gt_lit in  gt_operator.effects.literals:
                gt_p_name = f'{gt_lit.predicate.name}-{gt_lit.is_anti}-{gt_lit.is_negative}'
                if gt_p_name == p_name:
                    for v, gt_v in zip(lit.variables, gt_lit.variables):
                        param_mapping[v] = gt_v
    
    # Search over parametrizations using DP.
    def recurse(param_mapping):
        """Returns the best score and the corresponding parameter mapping, allowing a learned parameter to map to itself 
        if not matched to a ground-truth parameter."""
        memo = {}

        def helper(param_mapping):
            # Convert current mapping to a hashable key for memoization
            pm_key = frozenset(param_mapping.items())

            # Check memoization
            if pm_key in memo:
                return memo[pm_key]

            # Base case: all parameters are mapped
            if all(v in param_mapping for v in learned_operator.params):
                operator = deepcopy(learned_operator)
                # Apply the param mapping to both preconditions and effects
                for conj in [operator.effects.literals, operator.preconds.literals]:
                    for lit in conj:
                        lit.set_variables([param_mapping[v] for v in lit.variables])

                # Check if effects match
                if sorted(operator.effects.literals) != sorted(gt_operator.effects.literals):
                    memo[pm_key] = (-1, None)
                    return -1, None

                # Count how many preconditions match
                score = sum(lit in gt_operator.preconds.literals for lit in operator.preconds.literals)
                best_mapping = dict(param_mapping)
                memo[pm_key] = (score, best_mapping)
                return score, best_mapping

            # Recursive case: find an unmapped parameter in the learned operator
            unmapped = [v for v in learned_operator.params if v not in param_mapping]
            v = unmapped[0]

            used_vals = set(param_mapping.values())
            # Candidates: ground-truth parameters plus 'v' itself as a fallback
            next_i = len(gt_operator.params)
            not_assigned = True
            while not_assigned:
                not_assigned = False
                for param in used_vals:
                    if next_i == int(param._str.split(':')[0][len("?x"):]):
                        next_i += 1
                        not_assigned = True

            candidates = list(gt_operator.params) + [TypedEntity(f'?x{next_i}', Type(v._str.split(':')[1]))]

            best_score = -1
            best_mapping = None

            # Try all possible mappings for this parameter
            for c in candidates:
                if c not in used_vals:
                    param_mapping[v] = c
                    child_score, child_map = helper(param_mapping)
                    if child_score > best_score:
                        best_score = child_score
                        best_mapping = child_map
                    # Backtrack
                    del param_mapping[v]

            memo[pm_key] = (best_score, best_mapping)
            return best_score, best_mapping

        return helper(param_mapping)

    score, param_mapping = recurse(param_mapping)

    assert score != -1

    for conj in [learned_operator.effects.literals, learned_operator.preconds.literals]:
        for lit in conj:
            lit.set_variables([param_mapping[v] for v in lit.variables])
    learned_operator.params = set(param_mapping.values())
    return learned_operator