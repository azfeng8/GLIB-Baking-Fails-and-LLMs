from planning_modules.base_planner import PlannerTimeoutException, \
    NoPlanFoundException
from curiosity_modules import create_curiosity_module
from operator_learning_modules import create_operator_learning_module
from planning_modules import create_planning_module
from pddlgym.structs import Anti, State, Not, LiteralConjunction, ground_literal
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

        #  # Load the demos
        # with open('bakingrealistic_demonstrations.pkl', 'rb') as f:
        #     transitions = pickle.load(f)
        # self._operator_learning_module._transitions = transitions       

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

class InteractiveAgent(Agent):
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


        # Load the demos
        with open('bakingrealistic_demonstrations.pkl', 'rb') as f:
        # with open('transitions.pkl', 'rb') as f:
            transitions = pickle.load(f)
        self._operator_learning_module._transitions = transitions

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
        self._preconds_plan = []
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

        # # Load NDRs
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
        logging.info("Loaded subgoals.")
        # logging.info(self.subgoals)
    
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
    
    def _get_action_with_preconds_as_goals(self, state):
        """Returns the action, or None, if the stopping condition is reached.

        Stopping condition:
            If all of the preconditions are either unreachable from this state or the same preconditions has already had an action tried from it.
        """
        # Have successfully executed the plan to the operator preconds, and will execute the operator next
        if len(self._preconds_plan) == 1:
            self.finished_preconds_plan = True
            self.actions_since_last_subgoal = []
            logging.info(f"FOLLOWING PLAN: {self._preconds_plan}")
            self._ops_preconds_executed.add(self._op_preconds_to_execute)
            self._op_preconds_to_execute = None
            return self._preconds_plan.pop()

        # Follow plan to the operator's preconditions
        elif len(self._preconds_plan) > 0:
            self.finished_preconds_plan = False
            logging.info(f"FOLLOWING PLAN: {self._preconds_plan}")
            return self._preconds_plan.pop(0)

        # set a hyperparameter for how many ground preconditions to try.
        NUM_TRIES = 200

        action_predicates = set(p.name for p in self.action_space.predicates)
        for op in sorted(self.learned_operators, key=lambda op: op.name):
            # since the last time operators were learned, if operator has been successfully executed at the end of the plan, or
            # the plan failed in the middle to the operator preconditions, skip it.
            if op.name in self._ops_preconds_executed: continue
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
                if plan is not None:
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

        # check if plan is empty (goal is already satisfied in current state.)
        assignments = find_satisfying_assignments(state.literals, grounded_precond_lits, allow_redundant_variables=False)
        goal_is_satisfied_in_current_state = False
        for assignment in assignments:
            if all(var._str.split(':')[0] == val._str.split(':')[0] for var, val in assignment.items()):
                goal_is_satisfied_in_current_state = True
                break

        if goal_is_satisfied_in_current_state:
            return []

        # Use GLIB_G1 curiosity module to find a plan to the goal

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
            self._action_in_plan = False
            logging.info("Getting plan to precondition...")
            action = self._get_action_with_preconds_as_goals(state)
            if action is None:
               self._action_in_plan_to_preconds = False
               self.precondition_targeting = False 
            else:
                self._action_in_plan_to_preconds = True
                return action
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
[10] Target preconditions in a selected episode.

*** Utils ***
[5] Dump the transitions and operators.
[9] Evaluate operators.

*** MISC ***
[3] Execute a random action, observe it, and reset to the previous achieved subgoal.
"""
        try:
            option = int(input(option_str))
        except:
            option = None
        while option is None and option not in [0,2,3,4,5,6,7,8,9,10]:
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
                self._preconds_plan = []
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
            assignments = find_satisfying_assignments(next_state.literals, positive_lits, allow_redundant_variables=False)
            logging.info(assignments)
            if len(assignments) > 0:
                for assignment in assignments:
                    # Check that all object names in the state literals match the object names in the goal
                    if all(var._str.split(':')[0] == val._str.split(':')[0] for var, val in assignment.items()):
                        self.precondition_targeting = True
                        logging.info(f"ACHIEVED SUBGOAL {self.subgoals[self.next_subgoal_idx]}")
                        self.next_subgoal_idx += 1
                        self.action_seq.extend(self.actions_since_last_subgoal)
                        self.actions_since_last_subgoal = []
                        self._plan_to_next_subgoal = None
                        self._last_plan_to_next_subgoal = None
                        break

        
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
    
class DemonstrationsAgent(Agent):
    """An agent with initial demonstration data to each of the 4 train tasks."""
    def __init__(self, domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path:Optional[str]):
        super().__init__(domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, log_llm_path)
        
        self.name = 'demoagent'
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
        FILEPATHS = [
           '/home/catalan/GLIB-Baking-Fails-and-LLMs/realistic-baking/llm_plans/train/1/problem1.txt',
           '/home/catalan/GLIB-Baking-Fails-and-LLMs/realistic-baking/llm_plans/train/2/problem2.txt',
           '/home/catalan/GLIB-Baking-Fails-and-LLMs/realistic-baking/llm_plans/train/3/problem3.txt',
           '/home/catalan/GLIB-Baking-Fails-and-LLMs/realistic-baking/llm_plans/train/4/problem4.txt',
        ]
        for problem_i, filepath in enumerate(FILEPATHS):
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


def get_hashable_preconds_action(preconds):
    # Sort preconditions by alphabetical order of its string representation.
    strings = []
    for pre in preconds:
        strings.append(f'{pre.predicate.name}(' + ','.join(pre.pddl_variables()) + ')')
    s = ','.join(sorted(strings))
    return s
    
def dump_intermediate_state(agent:InteractiveAgent, fname='transitions.pkl'):
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
        pickle.dump(agent._rand_state, f)
