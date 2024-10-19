from settings import AgentConfig as ac
from pddlgym.parser import PDDLDomainParser, Operator
from pddlgym.structs import TypedEntity, ground_literal, LiteralConjunction, Literal
from ndr.learn import run_main_search as learn_ndrs
from ndr.learn import get_transition_likelihood, print_rule_set, iter_variable_names
from ndr.ndrs import NOISE_OUTCOME, NDR, NDRSet
from openai_interface import OpenAI_Model
from llm_parsing import LLM_PDDL_Parser, find_closing_paran, ops_equal

import re
import itertools
import numpy as np
import pickle
from collections import defaultdict
from typing import Dict, Iterable, Optional
import pddlgym
import logging
import os
from copy import deepcopy


class ZPKOperatorLearningModule:

    def __init__(self, planning_operators, learned_operators, domain_name):
        self._domain_name = domain_name
        self._planning_operators = planning_operators
        self._learned_operators = learned_operators
        self._transitions = defaultdict(list)
        self._seed = ac.seed
        self._rand_state = np.random.RandomState(seed=ac.seed)
        self._learning_on = True
        self._ndrs:Dict[pddlgym.structs.Predicate,NDRSet] = {}
        self._fits_all_data = defaultdict(bool)
        # Logging
        self._actions = []
        self._first_nonNOP_itrs = []
        self.skills_with_NOPS_only = set([p.name for p in ac.train_env.action_space.predicates])

    def observe(self, state, action, effects, itr, **kwargs):
        if not self._learning_on:
            return

        if (action.predicate.name in self.skills_with_NOPS_only) and len(effects) != 0:
            self.skills_with_NOPS_only.remove(action.predicate.name)
            self._first_nonNOP_itrs.append(itr)

        self._transitions[action.predicate].append((state.literals, action, effects))

        # Check whether we'll need to relearn
        # logging.info(self._fits_all_data)
        if self._fits_all_data[action.predicate]:
            ndr = self._ndrs[action.predicate]
            if not self._ndr_fits_data(ndr, state, action, effects):
                self._fits_all_data[action.predicate] = False

        # Logging
        self._actions.append(action)

    def learn(self, iter, **kwargs):

        if not self._learning_on:
            return False, False

        # Check whether we have NDRs that need to be relearned
        is_updated = False
        for action_predicate in self._fits_all_data:
            if not self._fits_all_data[action_predicate]:
                transition_for_action = self._transitions[action_predicate]
                
                # This is used to prioritize samples in the learning batch
                def get_batch_probs(data):
                    assert False, "Assumed off"
                    # Favor more recent data
                    p = np.log(np.arange(1, len(data)+1)) + 1e-5
                    # Downweight empty transitions
                    for i in range(len(p)):
                        if len(data[i][2]) == 0:
                            p[i] /= 2.
                    p = p / p.sum()
                    return p

                # Initialize from previous set?
                if action_predicate in self._ndrs and \
                    ac.zpk_initialize_from_previous_rule_set[self._domain_name]:
                    init_rule_sets = {action_predicate : self._ndrs[action_predicate]}
                else:
                    init_rule_sets = None

                # max explain_examples_transitions
                max_ee_transitions = ac.max_zpk_explain_examples_transitions[self._domain_name]

                learned_ndrs = learn_ndrs({action_predicate : transition_for_action},
                    max_timeout=ac.max_zpk_learning_time,
                    max_action_batch_size=ac.max_zpk_action_batch_size[self._domain_name],
                    get_batch_probs=get_batch_probs,
                    init_rule_sets=init_rule_sets,
                    rng=self._rand_state,
                    max_ee_transitions=max_ee_transitions,
                )
                ndrs_for_action = learned_ndrs[action_predicate]
                self._ndrs[action_predicate] = ndrs_for_action

                self._fits_all_data[action_predicate] = True
                is_updated = True 

        # Update all learned_operators
        if is_updated:
            self._planning_operators.clear()
            self._learned_operators.clear()
            for ndr_set in self._ndrs.values():
                i = 0
                for ndr in ndr_set:
                    # operator = ndr.determinize(name_suffix=i)
                    operators = ndr.multi_determinize(name_suffix=i)
                    i += len(operators)
                    # No point in adding an empty effect or noisy effect operator
                    for operator in operators:
                        if len(operator.effects.literals) == 0 or NOISE_OUTCOME in operator.effects.literals:
                            continue
                        self._planning_operators.add(operator)
                        self._learned_operators.add(operator)

            # print_rule_set(self._ndrs)

        return is_updated, is_updated

    def turn_off(self):
        self._learning_on = False

    def get_probability(self, transition):
        action = transition[1]
        if action.predicate not in self._ndrs:
            return 0.
        ndr_set = self._ndrs[action.predicate]
        selected_ndr = ndr_set.find_rule(transition)
        return get_transition_likelihood(transition, selected_ndr)

    def _ndr_fits_data(self, ndr:NDR, state, action, effects):
        prediction = ndr.predict_max(state, action)
        return sorted(prediction) == sorted(effects)
        # return abs(1 - self.get_probability((state.literals, action, effects))) < 1e-5

from pddlgym.inference import find_satisfying_assignments
from pddlgym.structs import ground_literal

class LLMZPKWarmStartOperatorLearningModule(ZPKOperatorLearningModule):
    """The ZPK operator learner but initialized with operators output by an LLM."""

    def __init__(self, planning_operators, learned_operators, domain_name, llm):
        """Loads the LLM-proposed operators and initializes.

        Args:
            planning_operators (set[Operator]): set of operators to use in planning for exploration (shared with other modules)
            learned_operators (set[Operator]): set of learned operators (shared with other modules)
            domain_name (str): pddlgym domain name
            llm (OpenAI_Model): LLM interface

        Raises:
            Exception: if the option to load operators is not one of the choices ['skill-conditioned', 'goal-conditioned', 'combined'], or the temperature provided is not in [0, 1]
        """
        super().__init__(planning_operators, learned_operators, domain_name)

        self._llm:OpenAI_Model = llm
        ap = {p.name: p for p in ac.train_env.action_space.predicates}
        op = {p.name: p for p in ac.train_env.observation_space.predicates}
        # Collect the object types in this domain.
        types = set()
        for p in (ac.train_env.action_space.predicates + ac.train_env.observation_space.predicates):
            for t in p.var_types:
                types.add(t)
        self._llm_parser = LLM_PDDL_Parser(ap, op, types)

        # Initialize the operators from the LLM.
        all_ops = []
        if ac.init_ops_method == 'goal-conditioned':
            dir = f'ada_init_operators/{self._domain_name}'
            file = 'manually_labeled_ops_fulltrainset.pkl'
            with open(os.path.join(dir, file), 'rb') as f:
                trainset_ops = pickle.load(f)
            for op_set in trainset_ops:
                for op in op_set:
                    all_ops.append(op)
        elif ac.init_ops_method == 'skill-conditioned':
            dir = f'todo_prompt_responses_temperature{ac.temperature}/{self._domain_name.lower()}_llm_responses'
            for file in os.listdir(dir):
                with open(os.path.join(dir, file), 'rb') as f:
                    response = pickle.load(f)[0]
                ops = self._llm_parser.parse_operators(response)
                for op in ops:
                    all_ops.append(op)
        elif ac.init_ops_method == 'skill-conditioned-two-stage':
            dir = f'skill_conditioned_2stage_temp{ac.temperature}/{self._domain_name}'
            file = 'ops.pkl'
            with open(os.path.join(dir, file), 'rb') as f:
                ops = pickle.load(f)
            all_ops.extend(ops)
        elif ac.init_ops_method == 'combined-todo-goal':
            dir = f'todo_prompt_responses_temperature{ac.temperature}/{self._domain_name.lower()}_llm_responses'
            for file in os.listdir(dir):
                with open(os.path.join(dir, file), 'rb') as f:
                    response = pickle.load(f)[0]
                ops = self._llm_parser.parse_operators(response)
                for op in ops:
                    all_ops.append(op)
            dir = f'ada_init_operators/{self._domain_name}'
            file = 'manually_labeled_ops_fulltrainset.pkl'
            with open(os.path.join(dir, file), 'rb') as f:
                trainset_ops = pickle.load(f)
            for op_set in trainset_ops:
                for op in op_set:
                    all_ops.append(op)
        elif ac.init_ops_method == 'combined-all':
            dir = f'todo_prompt_responses_temperature{ac.temperature}/{self._domain_name.lower()}_llm_responses'
            for file in os.listdir(dir):
                with open(os.path.join(dir, file), 'rb') as f:
                    response = pickle.load(f)[0]
                ops = self._llm_parser.parse_operators(response)
                for op in ops:
                    all_ops.append(op)
            dir = f'ada_init_operators/{self._domain_name}'
            file = 'manually_labeled_ops_fulltrainset.pkl'
            with open(os.path.join(dir, file), 'rb') as f:
                trainset_ops = pickle.load(f)
            for op_set in trainset_ops:
                for op in op_set:
                    all_ops.append(op)
            dir = f'skill_conditioned_2stage_temp{ac.temperature}/{self._domain_name}'
            file = 'ops.pkl'
            with open(os.path.join(dir, file), 'rb') as f:
                ops = pickle.load(f)
            all_ops.extend(ops)

        else:
            raise Exception(f'Not an option: {ac.init_ops_method}')

         # This tracks the most current version of the LLM-derived operators that should be used for GLIB.
        self._llm_ops = defaultdict(list)
        # This tracks all of the planning operators derived from the LLM tried so far.
        self._history_llm_ops = defaultdict(list)

        # Rename and remove duplicate operators from the loaded LLM operators. Add them to the history and currently used operators.
        for op in all_ops:
            action = [p for p in op.preconds.literals if p.predicate in ac.train_env.action_space.predicates][0]
            i = len(self._llm_ops[action.predicate])
            op.name = op.name.rstrip('0123456789') + "_" + action.predicate.name + "_" + str(i)
            not_equal = True
            for o in self._llm_ops[action.predicate]:
                if ops_equal(op, o):
                    not_equal = False
            if not_equal:
                self._llm_ops[action.predicate].append(op)
                self._history_llm_ops[action.predicate].append(op)
 
        # Update the planning operators with the LLM-derived operators
        for a in self._llm_ops:
            self._planning_operators.update(self._llm_ops[a])

        # A dict from operator name to the number of times it was executed in a plan and had no effects. This is continuously updated as planning operators are deleted / added.
        self._llm_op_fail_counts = defaultdict(lambda: 0)

        # State to say if the planning operators have changed from the observe() step.
        self._planning_ops_updated = False

    def observe(self, state, action, effects, itr, **kwargs):
        """Observe a transition.

        Args:
            state (pddlgym.structs.State): initial state of the transition
            action (Literal): action taken
            effects (set[Literal]): effects of the transition
            itr (int): training iteration #
        """
        if not self._learning_on:
            return

        # Add transition to training dataset
        self._transitions[action.predicate].append((state.literals, action, effects))

        # Logging info: get the first nonNOP
        if len(effects) != 0 and action.predicate.name in self.skills_with_NOPS_only:
            self.skills_with_NOPS_only.remove(action.predicate.name)
            self._first_nonNOP_itrs.append(itr)

        # Delete the LLM planning operators that don't hold.
        removes = []
        for op in self._llm_ops[action.predicate]:
            assignments = find_satisfying_assignments(list(state.literals) + [action], op.preconds.literals, allow_redundant_variables=False)

            # preconditions don't hold
            if len(assignments) == 0:
                continue

            # preconditions hold
            effects_hold = False            
            for assignment in assignments:
                full_assignments = find_satisfying_assignments(list(state.literals), op.effects.literals, init_assignments=assignment)
                for full_assignment in full_assignments:
                    pred_effects = set()
                    for l in op.effects.literals:
                        pred_effects.add(ground_literal(l, full_assignment))
                    if pred_effects == effects:
                    # if one of the ground effects of the operator matches the observed effects, then keep it. Otherwise, discard it.
                        effects_hold = True
            if not effects_hold:
                removes.append(op)

        if removes:
            self._planning_ops_updated = True
        else:
            self._planning_ops_updated = False

        for r in removes:
            self._llm_ops[action.predicate].remove(r)
            # Rename LLM ops
            if r.name in self._llm_op_fail_counts:
                del self._llm_op_fail_counts[r.name]
            self._planning_operators.remove(r)
            for i, op in enumerate(self._llm_ops[action.predicate]):
                op.name = op.name.rstrip('1234567890') + str(i)

        # Check whether we'll need to relearn
            # self._fits_all_data[action.predicate] is True once learned an initial NDR
        if self._fits_all_data[action.predicate]:
            ndr = self._ndrs[action.predicate]
            if not self._ndr_fits_data(ndr, state, action, effects):
                self._fits_all_data[action.predicate] = False
        # Logging
        self._actions.append(action)

    def learn(self, itr, skill_to_edit:Optional[tuple], **kwargs):
        """Updates the LLM-derived operators, calls LNDR, and updates the planning operators and learned operators.

        Args:
            itr (int): training iteration #.
            skill_to_edit (Optional[tuple[action predicate : pddlgym.structs.Predicate, operator_name : str]]): None if the operator was executed with effects. Otherwise, a tuple of the operator and skill executed in the plan which resulted in a No-Op.

        Returns:
            learned_ops_changed (bool)
            planning_ops_changed (bool)
        """
        if not self._learning_on:
            return False, False

        planning_ops_changed = False

        ### Update self._llm_ops
        
        op, action_pred, operator_name, same_precond_ops = None, None, None, None
        if skill_to_edit is not None:
            action_pred, operator_name = skill_to_edit
            # Only delete / modify the operator if it was derived from an LLM-proposed one (i.e., not from LNDR).
            for o in self._llm_ops[action_pred]:
                if o.name == operator_name:
                    op = o
                    break

        if op is not None:
            same_precond_ops = get_ops_with_same_preconds(op, self._llm_ops[action_pred])
            # increment fail count.
            self._llm_op_fail_counts[op.name] += 1

        # loop thru the operators with the same preconditions
        if same_precond_ops is not None:
            if ac.local_minima_method == 'delete-operator':
                planning_ops_changed = planning_ops_changed or self._delete_operator_update(same_precond_ops, action_pred)
            elif ac.local_minima_method == 'precond-relax':
                planning_ops_changed = planning_ops_changed or self._precond_relax_update(same_precond_ops, action_pred)
            else:
                raise Exception(f"Method {ac.local_minima_method} not found")

            for a in self._llm_ops:
                assert len(set([o.name for o in self._llm_ops[a]])) == len(self._llm_ops[a]), 'operator names are not all different'
        ################################################################################################################

        learned_ops_changed = False
        ### Call LNDR: Check whether we have NDRs that need to be relearned
        for action_predicate in self._fits_all_data:
            if not self._fits_all_data[action_predicate]:
                transition_for_action = self._transitions[action_predicate]
                
                # This is used to prioritize samples in the learning batch
                def get_batch_probs(data):
                    assert False, "Assumed off"

                # Initialize from previous set?
                if action_predicate in self._ndrs and \
                    ac.zpk_initialize_from_previous_rule_set[self._domain_name]:
                    init_rule_sets = {action_predicate : self._ndrs[action_predicate]}
                else:
                    init_rule_sets = None

                # max explain_examples_transitions
                max_ee_transitions = ac.max_zpk_explain_examples_transitions[self._domain_name]

                learned_ndrs = learn_ndrs({action_predicate : transition_for_action},
                    max_timeout=ac.max_zpk_learning_time,
                    max_action_batch_size=ac.max_zpk_action_batch_size[self._domain_name],
                    get_batch_probs=get_batch_probs,
                    init_rule_sets=init_rule_sets,
                    rng=self._rand_state,
                    max_ee_transitions=max_ee_transitions,
                )
                ndrs_for_action = learned_ndrs[action_predicate]
                self._ndrs[action_predicate] = ndrs_for_action

                self._fits_all_data[action_predicate] = True
                learned_ops_changed = True
        ################################################################################################################

        ### Update all learned and planning operators
        if planning_ops_changed or learned_ops_changed:
            self._planning_operators.clear()
            self._learned_operators.clear()

            for action_pred in self._llm_ops:
                self._planning_operators.update(self._llm_ops[action_pred])

            for act_pred in self._ndrs:
                for i, ndr in enumerate(self._ndrs[act_pred]):
                    suffix = len(self._llm_ops[act_pred]) + i
                    operator = ndr.determinize(name_suffix=suffix)
                    # No point in adding an empty effect or noisy effect operator
                    if len(operator.effects.literals) == 0 or NOISE_OUTCOME in operator.effects.literals:
                        continue
                    self._learned_operators.add(operator)
                    self._planning_operators.add(operator)

            # print_rule_set(self._ndrs)
        ################################################################################################################

        logging.info(f"LLM OPERATORS")
        for a in self._llm_ops:
            logging.info(f"Ops for action {a}: {[o.name for o in self._llm_ops[a]]}")

        # Need to add this to evaluate LLM operators on the first learning iteration (is_updated is False).
        if itr == 0:
            return True, True

        planning_ops_changed = planning_ops_changed or self._planning_ops_updated

        return learned_ops_changed, planning_ops_changed

    def _delete_operator_update(self, same_precond_ops:Iterable[Operator], action_pred):
        """Deletes the LLM-proposed operator if it fails more than AgentConfig.operator_fail_limit (see settings.py) times.

        Args:
            same_precond_ops: operators with the same precondition and skill as the operator that was executed in the plan with no effects.
            action_pred (pddlgym.structs.Predicate): skill that was executed in the plan with no effects.
        Returns:
            is_updated (bool): True if the operators changed, False otherwise.
        """
        is_updated = False
        for op in same_precond_ops:
            # if the operator has exceeded the hyperparam:
                if self._llm_op_fail_counts[op.name] > ac.operator_fail_limit:
                    self._llm_ops[action_pred].remove(op)
                    self._planning_operators.remove(op)
                    del self._llm_op_fail_counts[op.name]
                    logging.info(f"DELETED LLM OPERATOR: {op.name}")
                    # Maintain the naming convention of the LLM ops: the highest suffix number is less than the number of LLM planning operators with that skill
                        # To maintain, rename the operators after deleting one.
                    names_map = {}
                    for i, op in enumerate(self._llm_ops[action_pred]):
                        new_name = op.name.rstrip('1234567890') + str(i)
                        names_map[op.name] = new_name
                        op.name = new_name
                    # Maintain the operator fail counts indexed by operator name, by migrate the fail counts from old to new names.
                    new_fail_counts = defaultdict(lambda:0)
                    for op_name in self._llm_op_fail_counts:
                        if op_name in names_map:
                            new_fail_counts[names_map[op_name]] = self._llm_op_fail_counts[op_name]
                        else:
                            new_fail_counts[op_name] = self._llm_op_fail_counts[op_name]
                    self._llm_op_fail_counts = new_fail_counts
                    is_updated = True
        return is_updated
     
    def _precond_relax_update(self, same_precond_ops:Iterable[Operator], action_pred):
        """Relaxes the precondition of the LLM-derived operator if it fails more than AgentConfig.operator_fail_limit (see settings.py) times. If the precondition is empty, delete the operator.

        Args:
            same_precond_ops: operators with the same precondition and skill as the operator that was executed in the plan with no effects.
            action_pred (pddlgym.structs.Predicate): skill that was executed in the plan with no effects.
        Returns:
            is_updated: True if operators changed
        """
        is_updated = False
        for op in same_precond_ops:
            if self._llm_op_fail_counts[op.name] > ac.operator_fail_limit:
                ### Delete the operator
                self._llm_ops[action_pred].remove(op)
                del self._llm_op_fail_counts[op.name]
                self._planning_operators.remove(op)
                logging.info(f"DELETED LLM OPERATOR: {op.pddl_str()}")

                # Maintain the naming convention of the LLM ops: the highest suffix number is less than the number of LLM planning operators with that skill
                    # To maintain, rename the operators after deleting one.
 
                names_map = {}
                for i, operator in enumerate(self._llm_ops[action_pred]):
                    new_name = operator.name.rstrip('1234567890') + str(i)
                    names_map[operator.name] = new_name
                    operator.name = new_name

                # Maintain the operator fail counts indexed by operator name, by migrate the fail counts from old to new names.
                new_fail_counts = defaultdict(lambda:0)
                for op_name in self._llm_op_fail_counts:
                    if op_name in names_map:
                        new_fail_counts[names_map[op_name]] = self._llm_op_fail_counts[op_name]
                    else:
                        new_fail_counts[op_name] = self._llm_op_fail_counts[op_name]
                self._llm_op_fail_counts = new_fail_counts


                # if the precondition is not empty, generate a new operator for each literal in the precondition. Don't add operators already tried for planning from the LLM (i.e. allow duplicates with those planning operators from LNDR).
                # The precondition is empty if there is just the action predicate in the precondition.
                if len(op.preconds.literals) > 1:
                    preconds = deepcopy(op.preconds.literals)
                    for i, lit in enumerate(preconds):
                        if lit.predicate.name == action_pred.name:
                            action = lit
                            preconds.remove(action)
                    ### Add the new operators, avoiding duplicates from the history of LLM-derived operators.
                    for lit in preconds:
                        edited_preconds = deepcopy(preconds)
                        edited_preconds.remove(lit)
                        params = set()
                        for l in edited_preconds + op.effects.literals + [action]:
                            for v in l.variables:
                                params.add(v)
                        new_op = Operator(op.name, params, LiteralConjunction(edited_preconds + [action]), op.effects)
                        is_dup = False
                        for operator in self._history_llm_ops[action_pred]:
                            if ops_equal(operator, new_op):
                                is_dup = True
                        if not is_dup:
                            suffix = len(self._llm_ops[action_pred])
                            new_op.name = new_op.name.rstrip('0123456789') + str(suffix)
                            self._llm_ops[action_pred].append(new_op)
                            self._history_llm_ops[action_pred].append(new_op)
                            self._llm_op_fail_counts[new_op.name] = 0
                            logging.info(f"ADDED LLM OPERATOR: {new_op.pddl_str()}")
                is_updated = True
        return is_updated

    def _create_todo_prompt(self):
        """Generate the prompt using operator names, the action parameters (a subset of operator parameters), object types, and the observation predicates with types.
        """
        #Create the PDDL domain header without the extra syntax and action predicates that may confuse the LLM
        types = set()
        lines = []
        for p in ac.train_env.observation_space.predicates:
            types |= set(p.var_types)
            s = f"({p.name} " + " ".join(p.pddl_variables()) + ")"
            lines.append(s)
        predicates = "(:predicates\n\t\t" + "\n\t\t".join(lines) + "\n\t)"
        types = "(:types " + " ".join(types) + ")"
        header = f"(define (domain {self._domain_name.lower()})\n\t{types}\n\t{predicates}\n"

        prompt = "# Fill in the <TODO> to complete the PDDL domain.\n" + header  + "\n"
        for action_pred in ac.train_env.action_space.predicates:
            s = f"\t(:action {action_pred.name}\n"
            s += "\t\t:parameters (" + " ".join(action_pred.pddl_variables()) + " <TODO>)\n"
            s += f"\t\t:precondition (and\n\t\t\t\t<TODO>\n\t\t)\n"
            s += f"\t\t:effect (and\n\t\t\t\t<TODO>\n\t\t)\n\t)"
            prompt += s + "\n"
        prompt += ")"
        logging.info(f"Prompt: {prompt}")

        return prompt

    def _query_llm(self, prompt):
        """Do a network request to the LLM for a response of the TODO prompt.

        Args:
            prompt (str): TODO-template prompt for the LLM, of one operator per skill.

        Returns:
            str: response from the LLM
        """
        response, path = self._llm.sample_completions([{"role": "user", "content": prompt}], temperature=0, seed=self._seed, num_completions=1)
        response = response[0]
        logging.info(f"Got response {response}")
        logging.debug(f"Saved response at path: {path}")
        return response

def get_ops_with_same_preconds(op, ops) -> list[Operator]:
    """Get the set of operators with the same preconditions as the operator, agnostic to different parametrizations.

    Args:
        op (Operator): the operator with preconditions of interest
        ops (list[Operator]): operators all of the same skill, to find a subset of with the same preconditions as `op`.

    Returns:
        list[Operator]
    """
    same_preconds_ops = []
    op_counts = {}
    op_params = set()
    for lit in op.preconds.literals:
        op_counts.setdefault(lit.predicate.name, 0)
        op_counts[lit.predicate.name] += 1
        # Get a list of parameter names that exist in the preconditions
        for v in lit.variables:
            v_name = v._str.split(':')[0]
            op_params.add(v_name)
    op_params = list(op_params)
 
    for o in ops:
        # if the number of literals is different, continue
        if len(o.preconds.literals) != len(op.preconds.literals):
            continue
        counts = {}
        params = set()
        for lit in o.preconds.literals:
            counts.setdefault(lit.predicate.name, 0)
            counts[lit.predicate.name] += 1
            # Get the parameter names that exist in the preconditions
            for v in lit.variables:
                v_name = v._str.split(':')[0]
                params.add(v_name)

        # if the num predicates are different (count # of each predicate), continue
        if op_counts != counts:
            continue
            
        # Get a list of parameter names that exist in the preconditions
        params = list(params)

        # if the number of arguments in the preconditions don't match, continue
        if len(params) != len(op_params):
            continue
 
        # Get all permutations of them, using each as a new mapping to the list of param names for `op`
        for perm in itertools.permutations(params):
            # create the new set of literals with the new names.
            new_preconds = []
            names_map = dict(zip(perm, op_params))
            for lit in o.preconds.literals:
                args = []
                for v in lit.variables:
                    args.append(names_map[v._str.split(':')[0]])
                new_preconds.append(Literal(lit.predicate, args))
            
            # if the new set matches the set(op.preconds.literals):
            if set(new_preconds) == set(op.preconds.literals):
                # add the operator
                same_preconds_ops.append(o)
                break
    return same_preconds_ops



# # Debug code
# if __name__ == '__main__':
#     def get_batch_probs(data):
#         assert False, "Assumed off"
#         # Favor more recent data
#         p = np.log(np.arange(1, len(data)+1)) + 1e-5
#         # Downweight empty transitions
#         for i in range(len(p)):
#             if len(data[i][2]) == 0:
#                 p[i] /= 2.
#         p = p / p.sum()
#         return p


#     with open('/home/catalan/temp/cleanpan_data.pkl', 'rb') as f:
#         data = pickle.load(f)
#     with open('/home/catalan/temp/cleanpan_ndrs.pkl', 'rb') as f:
#         ndrs = pickle.load(f)
#     with open('/home/catalan/temp/cleanpan_pred.pkl', 'rb') as f:
#         pred = pickle.load(f)
#     ndrs = learn_ndrs({pred: data}, max_timeout=ac.max_zpk_learning_time, max_action_batch_size=ac.max_zpk_action_batch_size['Baking'], get_batch_probs=get_batch_probs,init_rule_sets=None, rng=np.random.RandomState(seed=ac.seed), max_ee_transitions= ac.max_zpk_explain_examples_transitions["Baking"])
#     for ndr in ndrs[pred]:
#         print(ndr.determinize())
    
#     nonnoop = []
#     for s,a,e in data:
#         if e != set():
#             nonnoop.append((s,a,e))
#     ndrs_nonnoop = learn_ndrs({pred: nonnoop}, max_timeout=ac.max_zpk_learning_time, max_action_batch_size=ac.max_zpk_action_batch_size['Baking'], get_batch_probs=get_batch_probs,init_rule_sets=None, rng=np.random.RandomState(seed=ac.seed), max_ee_transitions= ac.max_zpk_explain_examples_transitions["Baking"])
#     for ndr in ndrs_nonnoop[pred]:
#         print(ndr.determinize())