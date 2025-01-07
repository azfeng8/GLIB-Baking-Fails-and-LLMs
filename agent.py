from pprint import pprint
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
from settings import EnvConfig as ec
from settings import AgentConfig as ac
from ndr.learn import print_rule_set
from ndr.ndrs import NOISE_OUTCOME
from llm_parsing import LLM_PDDL_Parser
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
                 planning_module_name):
        """

        Args:
            domain_name (str): from PDDLGym environment
            action_space : from PDDLGym environment
            observation_space : from PDDLGym environment
            curiosity_module_name (str): 
            operator_learning_name (str): 
            planning_module_name (str): 
        """
        self.name = "agent"
        self.domain_name = domain_name
        self.curiosity_module_name = curiosity_module_name
        self.operator_learning_name = operator_learning_name
        self.planning_module_name = planning_module_name
        self._rand_state = np.random.RandomState(seed=ac.seed)
        # The main objective of the agent is to learn good operators
        self.learned_operators = set()
        self.action_space = action_space
        self.obs_space = observation_space


        # The operator learning module learns operators. It should update the
        # agent's learned operators set
        self._operator_learning_module = create_operator_learning_module(
            operator_learning_name, self.learned_operators, self.domain_name, self._rand_state)
        # The planning module uses the learned operators to plan at test time.
        self._planning_module = create_planning_module(
            planning_module_name, self.learned_operators, domain_name,
            action_space, observation_space)
        # The curiosity module dictates how actions are selected during training
        # It may use the learned operators to select actions
        self._curiosity_module = create_curiosity_module(
            curiosity_module_name, action_space, observation_space,
            self._planning_module, self.learned_operators,
            self._operator_learning_module, domain_name, self._rand_state)

    ## Training time methods
    def get_action(self, state, _problem_idx, _precond_targeting_only):
        """Get an exploratory action to collect more training data.
           Not used for testing. Planner is used for testing."""
        if self.domain_name == 'Bakinglarge' and 'oracle' not in self.curiosity_module_name:
            obs_literals = set()
            for lit in state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    obs_literals.add(lit)
            state = State(frozenset(obs_literals), state.objects, state.goal)

        start_time = time.time()
        in_plan, op_name, action = self._curiosity_module.get_action(state)
        logging.info(f"Getting action took {time.time() - start_time}")

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
        if self.domain_name == 'Bakinglarge':
            obs_literals = set()
            next_obs_literals = set()
            for lit in state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    obs_literals.add(lit)
            for lit in next_state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    next_obs_literals.add(lit)
            new_state = State(frozenset(obs_literals), state.objects, state.goal)
            new_next_state = State(frozenset(next_obs_literals), next_state.objects, next_state.goal)
            # Get effects
            new_effects = self._compute_effects(new_state, new_next_state)
        effects = self._compute_effects(state, next_state)
        logging.info(f"EFFECTS: \n{effects}")
        # Add data
        if self.domain_name == 'Bakinglarge':
            self._operator_learning_module.observe(new_state, action, new_effects)
        else:
            self._operator_learning_module.observe(state, action, effects)
        # Some curiosity modules might use transition data
        self._curiosity_module.observe(state, action, effects)
        self.episode_start = False

    def learn(self, itr):
        # Learn
        some_learned_operator_changed, _ = self._operator_learning_module.learn()

        # Used in LLMIterative only
        if self.operator_learning_name in ['LLM+LNDR', 'LLMIterative+LNDR']:
            self._curiosity_module.learn(itr)

        if some_learned_operator_changed:
            self._curiosity_module.learning_callback()
        # for k, v in self._operator_learning_module._ndrs.items():
        #     print(k)
        #     print(str(v))
        return some_learned_operator_changed, _

    def reset_episode(self, state):
        obs_literals = set()
        if self.domain_name == 'Bakinglarge':
            for lit in state.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    obs_literals.add(lit)
            state = State(frozenset(obs_literals), state.objects, state.goal)

        self._curiosity_module.reset_episode(state)

    @staticmethod
    def _compute_effects(state, next_state):
        positive_effects = {e for e in next_state.literals - state.literals}
        negative_effects = {Anti(ne) for ne in state.literals - next_state.literals}
        return positive_effects | negative_effects

    ## Test time methods
    def get_policy(self, problem_fname, use_learned_ops=False):
        """Get a plan given the learned operators and a PDDL problem file."""
        return self._planning_module.get_policy(problem_fname, use_learned_ops)


class DemonstrationsAgent(Agent):
     def __init__(self, domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name):
        super().__init__(domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name)
        self.name = 'demoagent'   

        # Load the demos
        demos_path = f'demonstrations/{self.domain_name.lower()}_demonstrations.pkl'
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
        self.finished_plan = False
 
        

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
        raise Exception("Done with demos. Should have terminated when prompted.")
            

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
    
 
class StudentAgent(Agent):
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
    #TODO: update docstring
    """

    # Number of lits to change in the goal.
    MAX_LIT_CHANGES = 3

    def __init__(self, domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name):
        super().__init__(domain_name, action_space, observation_space,
                         curiosity_module_name, operator_learning_name,
                         planning_module_name)
        self.name = 'student'

        # List of subgoals to achieve in order, each subgoal is a LiteralConjunction
        self.subgoals = []
        # Which subgoal index we are currently trying to achieve, or -np.inf as the null index
        self.next_subgoal_idx = -np.inf
        # Plan to the current subgoal
        self.plan_to_next_subgoal = None
        # The final action to execute once all subgoals are done
        self._final_action = None

        # Load the ground truth operators for the teacher policies
        if self.domain_name == 'Bakinglarge':
            domain_parser = PDDLDomainParser( '/home/catalan/GLIB-Baking-Fails-and-LLMs/realistic-baking/dom-parse-gt-operators.pddl')
            self._ground_truth_operators = {deepcopy(domain_parser.operators[o]) for o in domain_parser.operators}
        else:
            self._ground_truth_operators = {deepcopy(op) for op in ac.train_env.domain.operators.values()}

        # Get parser for operators from strings
        obj_types = set()
        for p in (self.action_space.predicates + self.obs_space.predicates):
            for t in p.var_types:
                obj_types.add(t)
        self.operator_parser = LLM_PDDL_Parser({p.name: p for p in self.action_space.predicates}, {p.name: p for p in self.obs_space.predicates}, obj_types)

        # Keep track of goals already tried
        self._visited_goals_states_teacher_mode = set()

        # Keep track if the plan has been finished, for environment resets. This flag is checked and reset in main.py.
        self.finished_plan = False

        # Turn off lifted goal babbling
        self._curiosity_module._ignore_mutex = False
        self._curiosity_module._ignore_statics = False
        self._curiosity_module._compute_goals = False
 
        # Remove special predicates used for PDDLGym simulation in the ground truth operators.
        if self.domain_name == 'Bakinglarge':
            for op in self._ground_truth_operators:
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
 
    
    def reset_episode(self, state):
        """Reset the episode and load subgoals from subgoals_file if provided."""
        # First do the StudentAgent's reset logic (which calls parent's reset too)
        super().reset_episode(state)

        # Clear our subgoal info
        self.subgoals = []
        self.next_subgoal_idx = -np.inf
        self.plan_to_next_subgoal = None
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

        self.subgoals = []
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

            self.finished_plan = True
            self.plan_to_next_subgoal = None
            self.next_subgoal_idx = -np.inf

            logging.info("All subgoals achieved. Executing final action.")
            # We don't plan for a single action; we just do it directly
            # (assuming the final action is guaranteed to be grounded).
            action = self._final_action
            self._final_action = None
            return action
        elif self.plan_to_next_subgoal is not None and len(self.plan_to_next_subgoal) == 0 and self.next_subgoal_idx == -np.inf:
            # Ground the action and return it
            return self._execute_plan(self.plan_to_next_subgoal, state)

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
                    ops=self._ground_truth_operators,
                    bakinglarge_file=True,
                    use_learned_ops=False
                )
            except NoPlanFoundException:
                logging.info("No plan found to subgoal.")
            except PlannerTimeoutException:
                logging.info("Planner timed out for subgoal.")
            finally:
                if os.path.exists(problem_fname):
                    os.remove(problem_fname)

            if plan is not None:
                logging.info(f"Found plan to subgoal {self.subgoals[self.next_subgoal_idx]}: {plan}")
                # if plan is empty, try next subgoal
                while plan == []:
                    self.next_subgoal_idx += 1
                    if self.next_subgoal_idx >= len(self.subgoals):
                        break
                    plan = self._get_ground_truth_plan(self.subgoals[self.next_subgoal_idx], state)
                    logging.info(f"Subgoal already achieved. FOUND PLAN to next subgoal {self.subgoals[self.next_subgoal_idx]}: {plan}")
                return self._execute_plan(plan, state)
            else:
                # Plan not found or timed out, you can do manual logic here
                # or just return None. Possibly reset plan, do user prompting, etc.
                logging.info(f"State:")
                for lit in sorted(state.literals):
                    logging.info(lit.pddl_str())
                raise Exception("Plan to subgoal failed/timed out. Probably a bug in domain or subgoals file.")
        else:
            # 4) choose an operator and try informative goals: if planner times out or too many informative goals in the change bank, then prompt user for subgoals list, like in StudentAgent.
            logging.info("=== Step 4) operator-based 'informative goals' approach (StudentAgent style) ===")

            # Keep track of which operator names have already been tried
            operator_names_tried = set()
            all_operator_names = {op.name for op in self.learned_operators}

            # We will attempt to create a small "change bank" of goals for each operator
            # by flipping or negating some preconditions, etc.
            while operator_names_tried != all_operator_names:

                self._skip_to_next_op = False
                # 4a) pick an untried operator
                chosen_op = None
                for op in self._rand_state.permutation(sorted(self.learned_operators, key=lambda o: o.name)):
                    if op.name not in operator_names_tried:
                        chosen_op = op
                        logging.info(f"Chosen operator: {chosen_op.pddl_str()}")
                        break
                action_pred = [l.predicate for l in chosen_op.preconds.literals if l.predicate in self.action_space.predicates][0]

                ops_to_consider = []
                for o in self.learned_operators:
                    if o.name == chosen_op.name: continue
                    a = [l.predicate for l in o.preconds.literals if l.predicate in self.action_space.predicates][0]           

                    if a == action_pred:
                        ops_to_consider.append(o)
    
                # Ask to join as many operators as possible.
                #TODO: Make a note of the logic of how to join the operators here, and that this step could probably be automated.
                while True:
                        
                    logging.info(f"Ops to consider:")
                    for o in ops_to_consider:
                        logging.info(o.pddl_str())
                    try:
                        prompt = f"" #TODO: describe the options in the prompt
                        if not (len(ops_to_consider) == 0 or 'use-stand-mixer' in chosen_op.name):
                            file = get_input_cached("File containing merged operator, qq, or d? ").strip()

                            if file == 'd':
                                dump_intermediate_state(self)
                                logging.info("Dumped state")
                            elif file == 'qq':
                                self._skip_to_next_op = True

                            while file not in ('q', 'qq') and not os.path.exists(file):
                                file = get_input_cached("File containing merged operator, qq, or d? ").strip()

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
                                    logging.info(f"parsed joined operator: {chosen_op.pddl_str()}")
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
                            #TODO: add more descriptive prompt
                            name = get_input_cached("g.t. operator name or 'q' to manually enter goal").strip()
                            names = {o.name for o in self._ground_truth_operators}
                            while name not in names and name != 'q':
                                name = get_input_cached("g.t. operator name or 'q' to manually enter goal").strip()                           
                            if name == 'q':
                                # 'qq' skips this operator, while 'q' retries combining and matching operators
                                plan = self._prompt_for_grounded_goal_and_plan(state, chosen_op)
                                if plan not in ('q', 'qq'):
                                    return plan
                                elif plan == 'qq':
                                    self._skip_to_next_op = True
                                    break
                                elif plan == 'd':
                                    dump_intermediate_state(self)
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
                    operator_names_tried.add(chosen_op.name)
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
                            self._skip_to_next_op = True
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
                            mark = get_hashable_lits(tuple(sorted(goal)))
                            if (mark, learned_operator.pddl_str()) in self._visited_goals_states_teacher_mode:
                                logging.info(f"Skipping goal: {goal}")
                                continue

                            goal_no_action = [l for l in goal if goal if l.predicate not in self.action_space.predicates]
                            vars_ = sorted({ v for lit in goal_no_action for v in lit.variables })
                            # add differents
                            if self.domain_name == 'Bakinglarge':
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
                                return self._execute_plan(plan, state)
                            elif plan == -1:
                                # no plan found.
                                # mark this goal as visited
                                self._visited_goals_states_teacher_mode.add((mark, learned_operator.pddl_str()))
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
                option = int(get_input_cached(option_str).strip())
            except:
                option = None
            while option is None or (option not in [9,11,12]):
                try:
                    if option == 5:
                        logging.info("Dumping state.")
                        dump_intermediate_state(self)
                    option = int(get_input_cached(option_str).strip())
                except:
                    option = None           
            self.option = option
            return None

    def _prompt_for_grounded_goal_and_plan(self, state, operator):
        #TODO: in the docstring, explain what the subgoal file looks like.
        
        # Extend the planner timeout, which is expected to be short, and reset it to the short planner timeout when planning is done.
        timeout = ac.planner_timeout 
        ac.planner_timeout = 400
        # provide the grounded goal file according to the lifted goal and then plan to it.
        while True:
            # if option is qq, then skip all goals for this operator.
            #TODO: provide more descriptive options
            goal_file = get_input_cached("Enter the subgoal file: ").strip()
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
                logging.info(f"Loaded subgoals: {goal_file}")
                plan = self._get_ground_truth_plan(self.subgoals[self.next_subgoal_idx], state)
                assert plan not in (None, -1), f"No plan found or timed out"
                logging.info(f"FOUND PLAN: {plan}")
                while plan == []:
                    self.next_subgoal_idx += 1
                    if self.next_subgoal_idx >= len(self.subgoals):
                        break
                    plan = self._get_ground_truth_plan(self.subgoals[self.next_subgoal_idx], state)
                    logging.info(f"Subgoal already achieved. FOUND PLAN to next subgoal: {plan}")
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

    def _get_ground_truth_plan(self, goal, state):
        problem_fname = self._curiosity_module._create_problem_pddl(
            state, goal, prefix='glibl_preconds')
        # Get a plan
        try:
            if self.domain_name == 'Bakinglarge':
                plan, _ = self._planning_module.get_plan(
                    problem_fname, use_cache=False, use_learned_ops=False, bakinglarge_file=True, ops=self._ground_truth_operators)
                os.remove(problem_fname)
                for step in plan:
                    if 'use-stand-mixer' in step.predicate.name:
                        corrected_plan = self._parse_corrected_plan_with_mixing(plan, state)
                        return corrected_plan
                return plan
 
            else:
                plan, _ = self._planning_module.get_plan(
                    problem_fname, use_cache=False, use_learned_ops=False, ops=self._ground_truth_operators)
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
 
    def _execute_plan(self, plan, state):

        self.plan_to_next_subgoal = plan

        goals, act, operator_str = self._current_goal_action_operator
        if len(self.plan_to_next_subgoal) == 0 and self.next_subgoal_idx >= len(self.subgoals):
            self.finished_plan = True
            self.plan_to_next_subgoal = None
 
            action =  self._final_action
            logging.info(f"Executing final action: {action}")
            self._final_action = None
            self.next_subgoal_idx = -np.inf
            return action

        elif len(self.plan_to_next_subgoal) == 0:
            # Sampled lifted goal
            ground_act = self._curiosity_module._sample_action_from_goal(goals, act,state, self._rand_state)
            if ground_act is None:
                #TODO: more descriptive prmopt
                ground_act_str = get_input_cached("Grounding failed. Enter the grounded action or Ctrl-C to end: ").strip()
                while True:
                    try:
                        line = ground_act_str
                        if line.startswith("(") and line.endswith(")"):
                                line = line[1:-1]
                        items = line.split()
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
                        ground_act = act_pred(*typed_objs)
                        break
                    except Exception as e:
                        print(e)
                        traceback.print_exc() 
                        input("Continue or Ctrl-C to quit:")
                        continue
            mark = get_hashable_lits(tuple(sorted(goals)))
            self._visited_goals_teacher_mode.add((mark, operator_str))
            self.plan_to_next_subgoal = None
            self.finished_plan = True
            logging.info(f"Executing grounded action: {ground_act}")
            return ground_act

        return self.plan_to_next_subgoal.pop(0)

    def _parse_corrected_plan_with_mixing(self, plan, state):
        logging.info("Planner found this plan:")
        for step in plan:
            logging.info(f'{step.pddl_str()}')
        while True:
            try:
                # TODO: more descriptive prompt
                file = get_input_cached("Enter the corrected plan file: ").strip()
                # parse corrected plan.
                with open(file, 'r') as f:
                    lines = f.readlines()
                corrected_plan = []
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    if line.startswith("(") and line.endswith(")"):
                        line = line[1:-1]
                    items = line.split()
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
                    corrected_plan.append(act_pred(*typed_objs))

                logging.info(f"Parsed corrected plan from: {file}: {corrected_plan}")
                return corrected_plan
            except Exception as e:
                print(e)
                traceback.print_exc() 
                input("Continue or Ctrl-C to quit:")
                continue


 
    def observe(self, state, action, next_state, itr):
        """
        Overridden observe to handle:
            - If the plan fails (no effects), reset plan variables.
            - If subgoal is achieved, increment subgoal index, etc.

        Otherwise, reuse the parent's logic from StudentAgent.
        
        """
        if self.domain_name.lower() == 'Bakinglarge':
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
 
        # Check if the action had no effects => plan might be failing
        effects = self._compute_effects(state, next_state)
        logging.info(f"EFFECTS: \n{effects}")
        self._operator_learning_module.observe(state, action, effects)
        self._curiosity_module.observe(state, action, effects)

        if len(effects) == 0:
            logging.info("Plan execution failed mid-subgoal: resetting subgoal plan and subgoals.")
            self.plan_to_next_subgoal = None
            self.subgoals = []
            self.next_subgoal_idx = -np.inf
            self._current_goal_action_operator = None
            return True

        # Check if we have achieved the current subgoal
        # (only if we are in the middle of planning to subgoals)
        if self.next_subgoal_idx != -np.inf and self.next_subgoal_idx < len(self.subgoals):
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

def get_hashable_lits(lits):
    # Sort preconditions by alphabetical order of its string representation.
    strings = []
    for pre in lits:
        if pre.is_negative:
            pred = f'NOT-{pre.predicate.name}'
        else:
            pred = pre.predicate.name
        strings.append(f'({pred}' + ','.join(pre.pddl_variables()) + ')')
    s = ','.join(sorted(strings))
    return s
    
def dump_intermediate_state(agent:StudentAgent, fname='transitions.pkl'):
    with open(fname, 'wb') as f:
        pickle.dump(agent._operator_learning_module._transitions, f)
    with open('ops.pkl', 'wb') as f:
        pickle.dump(agent.learned_operators, f)
    with open('visited_preconds.pkl', 'wb') as f:
        pickle.dump(agent._visited_preconds_states_teacher_mode, f)
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

def get_input_cached(prompt):
    try:
        inp = next(ac.input_generator)
        logging.info(prompt)
        logging.info(f"Using cached input: {inp}")
        return inp
    except StopIteration:
        return input(prompt)