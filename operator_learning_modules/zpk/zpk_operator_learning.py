from settings import AgentConfig as ac
from pddlgym.parser import PDDLDomainParser, Operator
from pddlgym.structs import TypedEntity, ground_literal
from ndr.learn import run_main_search as learn_ndrs
from ndr.learn import get_transition_likelihood, print_rule_set, iter_variable_names
from ndr.ndrs import NOISE_OUTCOME, NDR, NDRSet
from openai_interface import OpenAI_Model
from llm_parsing import LLM_PDDL_Parser, find_closing_paran

import re
import numpy as np
import pickle
from collections import defaultdict
from typing import Dict
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
        logging.info(self._fits_all_data)
        if self._fits_all_data[action.predicate]:
            ndr = self._ndrs[action.predicate]
            if not self._ndr_fits_data(ndr, state, action, effects):
                self._fits_all_data[action.predicate] = False

        # Logging
        self._actions.append(action)

    def learn(self, iter=-1):

        if not self._learning_on:
            return False

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
                for i, ndr in enumerate(ndr_set):
                    operator = ndr.determinize(name_suffix=i)
                    # No point in adding an empty effect or noisy effect operator
                    if len(operator.effects.literals) == 0 or NOISE_OUTCOME in operator.effects.literals:
                        continue
                    self._planning_operators.add(operator)
                    self._learned_operators.add(operator)

            # print_rule_set(self._ndrs)

        return is_updated

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


class LLMZPKWarmStartOperatorLearningModule(ZPKOperatorLearningModule):
    """The ZPK operator learner but initialized with operators output by an LLM."""

    def __init__(self, planning_operators, learned_operators, domain_name, llm):
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

        # # Initialize the operators from the LLM.
        prompt = self._create_todo_prompt()
        llm_output = self._query_llm(prompt)
        operators = self._llm_output_to_operators(llm_output)
        self._llm_ops = defaultdict(list)
        for op in operators:
            action_pred = [l.predicate for l in op.preconds.literals if l.predicate in ac.train_env.action_space.predicates][0]
            self._llm_ops[action_pred].append(op)
        self._planning_operators.update(operators)
        self._evaluate_first_iteration = True

    def observe(self, state, action, effects, itr, **kwargs):
        if not self._learning_on:
            return

        self._transitions[action.predicate].append((state.literals, action, effects))

        if len(effects) != 0 and action.predicate.name in self.skills_with_NOPS_only:
            self.skills_with_NOPS_only.remove(action.predicate.name)
            self._first_nonNOP_itrs.append(itr)

        # Check whether we'll need to relearn
            # self._fits_all_data[action.predicate] is True once learned an initial NDR
        if self._fits_all_data[action.predicate] and action.predicate.name not in self.skills_with_NOPS_only:
            ndr = self._ndrs[action.predicate]
            if not self._ndr_fits_data(ndr, state, action, effects):
                self._fits_all_data[action.predicate] = False
        # Logging
        self._actions.append(action)

    def learn(self, itr=-1):
        """Only call LNDR on skills that have a nonNOP.
        This is justified since if the NDR has no effects, no operator is conceived. Thus, until a nonNOP is received, no operator is conceived.

        So, the LLM operator (if it exists) will be used in substitution.
        """
        if not self._learning_on:
            return False

        # Check whether we have NDRs that need to be relearned
        is_updated = False
        for action_predicate in self._fits_all_data:
            if action_predicate.name in self.skills_with_NOPS_only:
                continue
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
            for action_pred in self._ndrs:
                for i, ndr in enumerate(self._ndrs[action_pred]):
                    suffix = i + len(self._llm_ops[action_pred])
                    operator = ndr.determinize(name_suffix=suffix)
                    # No point in adding an empty effect or noisy effect operator
                    if len(operator.effects.literals) == 0 or NOISE_OUTCOME in operator.effects.literals:
                        continue
                    self._learned_operators.add(operator)
                    self._planning_operators.add(operator)
            for action_pred in self._llm_ops:
                if action_pred.name in self.skills_with_NOPS_only:
                    self._planning_operators.update(self._llm_ops[action_pred])

            # print_rule_set(self._ndrs)

        if self._evaluate_first_iteration and itr == 0:
            return True

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
        # response, path = self._llm.sample_completions([{"role": "user", "content": prompt}], temperature=0, seed=self._seed, num_completions=1)
        # response = response[0]
        with open(f'{self._domain_name.lower()}_llm_responses/{str(ac.seed)[-1]}.pkl', 'rb') as f:
            response = pickle.load(f)[0]
        logging.info(f"Got response {response}")
        # logging.debug(f"Saved response at path: {path}")
        return response


    def _llm_output_to_operators(self, llm_output) -> list[Operator]:
        """Parse the LLM output."""

        # Split the response into chunks separated by "(:action ", and discard the first chunk with the header and LLM chat.
        operator_matches = list(re.finditer("\(\:action\s", llm_output))
        operators = []
        end = len(llm_output)
        for match in operator_matches[::-1]: # Read matches from end of file to top of file
            start = match.start()
            operator_str = find_closing_paran(llm_output[start:end])
            ops = self._llm_parser.parse_operators(operator_str)
            if ops is None:
                continue
            for o in ops:
                if o is None: continue
                operators.append(o)
            end = match.start()

        return operators

def ops_equal(op1:Operator, op2:Operator):
    if op1.params != op2.params:
        return False
    if set(op1.preconds.literals) != set(op2.preconds.literals):
        return False
    if set(op1.effects.literals) != set(op2.effects.literals):
        return False
    return True

# Debug code
if __name__ == '__main__':
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


    with open('/home/catalan/temp/cleanpan_data.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('/home/catalan/temp/cleanpan_ndrs.pkl', 'rb') as f:
        ndrs = pickle.load(f)
    with open('/home/catalan/temp/cleanpan_pred.pkl', 'rb') as f:
        pred = pickle.load(f)
    ndrs = learn_ndrs({pred: data}, max_timeout=ac.max_zpk_learning_time, max_action_batch_size=ac.max_zpk_action_batch_size['Baking'], get_batch_probs=get_batch_probs,init_rule_sets=None, rng=np.random.RandomState(seed=ac.seed), max_ee_transitions= ac.max_zpk_explain_examples_transitions["Baking"])
    for ndr in ndrs[pred]:
        print(ndr.determinize())
    
    nonnoop = []
    for s,a,e in data:
        if e != set():
            nonnoop.append((s,a,e))
    ndrs_nonnoop = learn_ndrs({pred: nonnoop}, max_timeout=ac.max_zpk_learning_time, max_action_batch_size=ac.max_zpk_action_batch_size['Baking'], get_batch_probs=get_batch_probs,init_rule_sets=None, rng=np.random.RandomState(seed=ac.seed), max_ee_transitions= ac.max_zpk_explain_examples_transitions["Baking"])
    for ndr in ndrs_nonnoop[pred]:
        print(ndr.determinize())