import sys
sys.path.append("/home/catalan/GLIB-Baking-Fails-and-LLMs")
import numpy as np
import pickle
from settings import AgentConfig as ac
from pddlgym.parser import PDDLDomainParser
from pddlgym.structs import TypedEntity, ground_literal
from ndr.learn import run_main_search as learn_ndrs
from ndr.learn import get_transition_likelihood, print_rule_set, iter_variable_names
from ndr.ndrs import NOISE_OUTCOME, NDR, NDRSet
from openai_interface import OpenAI_Model
import re

from collections import defaultdict
from typing import Dict
import tempfile
import pddlgym


class ZPKOperatorLearningModule:

    def __init__(self, learned_operators, domain_name, planning_ops):
        self._domain_name = domain_name
        self._learned_operators = learned_operators
        self._planning_operators = planning_ops
        self._transitions = defaultdict(list)
        self._seed = ac.seed
        self._rand_state = np.random.RandomState(seed=ac.seed)
        self._learning_on = True
        self._ndrs:Dict[pddlgym.structs.Predicate,NDRSet] = {}
        self._fits_all_data = defaultdict(bool)

    def observe(self, state, action, effects, **kwargs):
        if not self._learning_on:
            return
        self._transitions[action.predicate].append((state.literals, action, effects))

        # Check whether we'll need to relearn
        if self._fits_all_data[action.predicate]:
            ndr = self._ndrs[action.predicate]
            if not self._ndr_fits_data(ndr, state, action, effects):
                self._fits_all_data[action.predicate] = False

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
                # if action_predicate.name == 'cleanpan':
                #     print(self._ndrs[action_predicate])
                #     with open('/home/catalan/temp/cleanpan_data.pkl', 'wb') as f:
                #         pickle.dump(self._transitions[action_predicate], f)
                #     with open('/home/catalan/temp/cleanpan_ndrs.pkl', 'wb') as f:
                #         pickle.dump(self._ndrs[action_predicate], f)
                #     with open('/home/catalan/temp/cleanpan_pred.pkl', 'wb') as f:
                #         pickle.dump(action_predicate, f)
                self._fits_all_data[action_predicate] = True
                is_updated = True 

                # Update planning operators for the action predicate here.
                removes = set()
                for op in self._planning_operators:
                    if action_predicate.name in op.name:
                        removes.add(op)
                for op in removes:
                    self._planning_operators.remove(op)
                for i,ndr in enumerate(ndrs_for_action):
                    operator = ndr.determinize(name_suffix=i)
                    if len(operator.effects.literals) == 0 or NOISE_OUTCOME in operator.effects.literals:
                        continue
                    self._planning_operators.add(operator)

        # Update all learned_operators
        if is_updated:
            self._learned_operators.clear()
            for ndr_set in self._ndrs.values():
                for i, ndr in enumerate(ndr_set):
                    operator = ndr.determinize(name_suffix=i)
                    # No point in adding an empty effect or noisy effect operator
                    if len(operator.effects.literals) == 0 or NOISE_OUTCOME in operator.effects.literals:
                        continue
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


class LLMZPKOperatorLearningModule(ZPKOperatorLearningModule):
    """The ZPK operator learner but initialized with operators output by an LLM."""

    def __init__(self, learned_operators, domain_name, llm):
        super().__init__(learned_operators, domain_name)

        # # Initialize the dataset with fake transitions induced from the LLM.
        # # Then call learning with this dataset.
        # # If these transitions are bad, they will eventually be treated as noise.
        # # If they are good, they will lead to nice initial operators.
        # prompt = self._create_prompt()
        # llm_output = self._query_llm(prompt)
        # operators = self._llm_output_to_operators(llm_output)
        # self._learned_operators.update(operators)
        # # Also need to initialize ndrs!
        # for op in operators:
        #     action = [p for p in op.preconds.literals
        #                 if p.predicate in ac.train_env.action_space.predicates][0]
        #     state_lits = frozenset(op.preconds.literals) - {action}
        #     effects = frozenset(op.effects.literals)
        #     # TODO: Rename the variables for clarity.
        #     self._transitions[action.predicate].append((state_lits, action, effects))
        #     self._fits_all_data[action.predicate] = False
        # self.learn()

        self._llm:OpenAI_Model = llm

        # # Initialize the operators from the LLM.
        prompt = self._create_prompt()
        llm_output = self._query_llm(prompt)
        operators = self._llm_output_to_operators(llm_output)
        self._learned_operators.update(operators)
        # Also need to initialize ndrs!
        for op in operators:
            # In initializing the learner from previous, we assume a
            # standard variable naming scheme.
            action = [p for p in op.preconds.literals
                        if p.predicate in ac.train_env.action_space.predicates][0]
            preconditions = sorted(set(op.preconds.literals) - {action})
            effects = list(op.effects.literals)
            variables = list(action.variables)
            for lit in preconditions + op.effects.literals:
                for v in lit.variables:
                    if v not in variables:
                        variables.append(v)
            sub = {old: TypedEntity(new_name, old.var_type)
                    for old, new_name in zip(variables, iter_variable_names())}
            action = ground_literal(action, sub)
            preconditions = [ground_literal(l, sub) for l in preconditions]
            effects = [ground_literal(l, sub) for l in effects]
            ndr = NDR(action, preconditions, np.array([1.0, 0.0]), [effects, [NOISE_OUTCOME]])
            ndrs = NDRSet(action, [ndr])
            self._ndrs[action.predicate] = ndrs

    def _create_header(self):
        """Create the PDDL Domain header."""
        env = ac.train_env
        preds = [p for p in env.action_space.predicates] + [p for p in env.observation_space.predicates]
        types = set()
        lines = []
        for p in preds:
            types |= set(p.var_types)
            s = f"({p.name} " + " ".join(p.pddl_variables()) + ")"
            lines.append(s)
        predicates = "(:predicates\n" + "\n".join(lines) + ")"
        actions = '; (:actions ' + ' '.join([p.name for p in env.action_space.predicates]) + ")\n"
        types = "(:types " + " ".join(types) + ")"

        return f"(define (domain {self._domain_name.lower()})\n(:requirements :typing)\n{types}\n{predicates}\n\n{actions}"

    def _create_prompt(self):
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
        print("Prompt:", prompt)

        return prompt

    def _query_llm(self, prompt):
        # response = self._llm.sample_completions([{"role": "user", "content": prompt}], temperature=0, seed=self._seed, num_completions=1)[0]
        response = """(define (domain baking)
        (:types ingredient oven pan soap)
        (:predicates
                (hypothetical ?v0 - ingredient)
                (inoven ?v0 - pan ?v1 - oven)
                (inpan ?v0 - ingredient ?v1 - pan)
                (iscake ?v0 - ingredient)
                (isegg ?v0 - ingredient)
                (isflour ?v0 - ingredient)
                (ismixed ?v0 - pan)
                (issouffle ?v0 - ingredient)
                (ovenisfull ?v0 - oven)
                (panhasegg ?v0 - pan)
                (panhasflour ?v0 - pan)
                (paninoven ?v0 - pan)
                (panisclean ?v0 - pan)
                (soapconsumed ?v0 - soap)
        )

        (:action bakecake
                :parameters (?v0 - ingredient ?v1 - oven)
                :precondition (and
                                (panhasegg ?v0 - ?v1)
                )
                :effect (and
                                (iscake ?v0)
                )
        )
        (:action bakesouffle
                :parameters (?v0 - ingredient ?v1 - oven)
                :precondition (and
                                (inpan ?v0 - ?v2)
                                (ismixed ?v2)
                                (paninoven ?v2 - ?v1)
                )
                :effect (and
                                (issouffle ?v0)
                )
        )
        (:action cleanpan
                :parameters (?v0 - pan ?v1 - soap)
                :precondition (and
                                (panisclean ?v0)
                                (soapconsumed ?v1)
                )
                :effect (and
                                (not (panisclean ?v0))
                )
        )
        (:action mix
                :parameters (?v0 - pan)
                :precondition (and
                                (inpan ?v1 - ?v0)
                                (panhasegg ?v1 - ?v0)
                                (panhasflour ?v1 - ?v0)
                )
                :effect (and
                                (ismixed ?v0)
                )
        )
        (:action putegginpan
                :parameters (?v0 - ingredient ?v1 - pan)
                :precondition (and
                                (isegg ?v0)
                                (panisclean ?v1)
                )
                :effect (and
                                (panhasegg ?v0 - ?v1)
                )
        )
        (:action putflourinpan
                :parameters (?v0 - ingredient ?v1 - pan)
                :precondition (and
                                (isflour ?v0)
                                (panisclean ?v1)
                )
                :effect (and
                                (panhasflour ?v0 - ?v1)
                )
        )
        (:action putpaninoven
                :parameters (?v0 - pan ?v1 - oven)
                :precondition (and
                                (inpan ?v0 - ?v2)
                                (ismixed ?v2)
                                (ovenisfull ?v1)
                )
                :effect (and
                                (inoven ?v0 - ?v1)
                )
        )
        (:action removepanfromoven
                :parameters (?v0 - pan)
                :precondition (and
                                (inoven ?v0 - ?v1)
                )
                :effect (and
                                (not (inoven ?v0 - ?v1))
                )
        )
)"""
        print("Got response", response)
        return response
    
    def _check_and_fix_operator_string(self, types, operator_str):
        """
        Parses the PDDL operator from the string and returns False if the operator string is malformed and can't be fixed, or the fixed string.

        Fixes:
           - Insert the action predicate into the preconditions as needed by PDDLGym.
           - If the predicate's arguments is a superset of the actual arguments, then drop the extra arguments.

        Throws:
            Exception if fixing went wrong.

        Args:
            types (set[str]): set of object types allowed in this domain
            operator_str (str): string containing a PDDL operator definition
        Returns:
            [str, bool], str: returns False, error_msg if the string is malformed. Returns the action_predicate if no errors detected.
            
        """
        action_name = re.search("[\w]+", operator_str).group(0)
        action_preds = {p.name:p for p in ac.train_env.action_space.predicates}
        if action_name not in action_preds:
            return False,f"Invalid operator found: {action_name}"
        # assert action_name in action_preds, f"Invalid action found: {action_name}"
        parameter_str = re.search("\:parameters[^\)]*\)", operator_str).group(0)
        pddl_variables = re.findall("\?[\w\d\s\-]+", parameter_str)
        param_names = []
        param_types = []
        for var_str in pddl_variables:
            var_str = var_str.strip()
            param_name = re.search("\?[\w\d]+", var_str).group(0)[1:] # trim the ? in beginning
            var_type = re.search("\-\s[\w]+", var_str).group(0)[2:] # trim the "- " in the beginning
            param_names.append(param_name)
            param_types.append(var_type)
            if var_type not in types:
                return False, f"type not valid: {var_type} in parameters: {parameter_str}. accepted types: {types}"
        # Make immutable
        param_names:tuple = tuple(param_names)
        param_types:tuple = tuple(param_types)

        # Check each precondition and effect have 1. valid predicate name, 2.predicate argument names match params, 3. predicate argument types match params 
        obs_preds = {o.name:o for o in ac.train_env.observation_space.predicates}
        precond_str_match = re.search(":precondition([\s\S]*?):effect", operator_str)
        precond_str_start, precond_str_end = (precond_str_match.start(), precond_str_match.end() - len(":effect"))
        precond_str = precond_str_match.group(0)[:-7] # trim the ":effect" at the end
        effect_str_match = re.search(":effect([\s\S]*?)\s\)", operator_str)
        effect_str_start, effect_str_end = (effect_str_match.start(), effect_str_match.end())
        effect_str = effect_str_match.group(0)
        # Contains 2 strings: precondition and effect parts
        edited_parts = []

        # The same editing happens for clauses in preconditions and effects
        for pre_or_eff in [precond_str, effect_str]:

            edited_predicate_strings:list[tuple[str,int,int]] = []

            for predicate_str_match in re.finditer("\([\w]+[\s\?\w-]*\)", pre_or_eff):
                predicate_str = predicate_str_match.group(0)
                if "and" in predicate_str or "not" in predicate_str: continue
                pred_name = re.search("[\w]+", predicate_str).group(0)
                if pred_name not in obs_preds:
                    # Drop the predicate if there are still predicates left in this precondition or effect
                    edited_predicate_strings.append(("", predicate_str_match.start(), predicate_str_match.end()))
                    continue
                arg_types = []
                arg_names = []
                for arg_name in re.findall("\?[\w\d]+", predicate_str):
                    arg_name = arg_name[1:] # remove the "?" in front
                    if arg_name not in param_names:
                        return False,f"Argument for {pred_name} in {pre_or_eff} not in parameters: {arg_name}"
                    arg_names.append(arg_name)
                    arg_types.append(param_types[param_names.index(arg_name)])

                # Collect the arguments of the predicate in order
                args = []
                types_dont_match = False
                for arg_type in obs_preds[pred_name].var_types:
                    if arg_type not in arg_types:
                        # Drop the predicate if there are still predicates left in this precondition or effect
                        edited_predicate_strings.append(("", predicate_str_match.start(), predicate_str_match.end()))
                        types_dont_match = True
                        break
                    i = arg_types.index(arg_type)
                    arg_types.pop(i)
                    args.append("?" + arg_names.pop(i))
                if types_dont_match:
                    continue
                # Edit the operator string: drop the extra arguments in this predicate, if any
                edited_pred_str = f"({pred_name} " + " ".join(args) + ")"
                edited_predicate_strings.append((edited_pred_str, predicate_str_match.start(), predicate_str_match.end()))

            empty_pre_or_eff:bool = True
            for p, _, _ in edited_predicate_strings:
                if p != "": empty_pre_or_eff = False

            if empty_pre_or_eff or len(edited_predicate_strings) == 0:
                return False, f"No valid predicates parsed for operator: {action_name}."

            # Cut and paste the edited parts back into the original string of the precondition or effect part.
            edited_pre_or_eff = ""
            i = 0
            for edited_s, start_i, end_i in edited_predicate_strings:
                edited_pre_or_eff += pre_or_eff[i:start_i] + edited_s
                i = end_i
            edited_pre_or_eff += pre_or_eff[i:]

            edited_parts.append(edited_pre_or_eff)

        # Cut and paste the edited parts back into the original string of the whole operator.
        operator_str = operator_str[:precond_str_start] + edited_parts[0] + operator_str[precond_str_end:effect_str_start] + edited_parts[1] + operator_str[effect_str_end:]

        # Insert the action predicate
        action_pred_types:list[str] = action_preds[action_name].var_types
        action_pred = f"\n\t\t\t\t({action_name} "
        p_names, p_types = list(param_names), list(param_types)
        for t in action_pred_types:
            if t not in p_types:
                return False,f"Action parameter of type {t} must be a subset of the LLM's generated parameters. {action_pred_types}, but found: {param_types}"
            i = p_types.index(t)
            p_types.pop(i)
            action_pred += f"?{p_names.pop(i)} "
        action_pred += ")\n"

        word_match = re.search("\([\w]+[\s\?\w]*", operator_str)
        word = word_match.group(0).strip()
        if word == "(and":
            begin = word_match.end()
            operator_str = "(:action " + operator_str[:begin] +  f"\t\t\t\t{action_pred}\t\t\t\t" + operator_str[begin:]
        elif word == "(not":
            begin = word_match.start()
            precond_end = re.search("\)[\s]*:effect", operator_str).start()
            operator_str = f"(:action " + operator_str[:begin] + f"(and \t\t\t\t{action_pred}\t\t\t\t" + operator_str[begin:precond_end] + ")" + operator_str[precond_end:]
        else:
            raise Exception(f"got word {word} in str while inserting action pred into: {operator_str}")
                
        return operator_str, ""

    def _llm_output_to_operators(self, llm_output):
        # Parse the LLM output using PDDLGym.
        
        # Automatically detect and correct malformed LLM output.

        # Split the response into chunks separated by "(:action ", and discard the first chunk with the header and LLM chat.
        operator_matches = list(re.finditer("\(\:action\s", llm_output))

        action_pddls = []

        # Collect the object types in this domain.
        types = set()
        for p in (ac.train_env.action_space.predicates + ac.train_env.observation_space.predicates):
            for t in p.var_types:
                types.add(t)

        end = len(llm_output)
        # Check each action's preconditions, parameters, and effects for valid names and consistent typing.
        for match in operator_matches[::-1]: # Read matches from end of file to top of file
            start = match.end()
            operator_str = llm_output[start:end]

            # Count parantheses: look for the closing to "(:action" to get the operator string.
            open_parans = 0
            close = 0
            i = start
            for c in operator_str:
                if close > open_parans:
                    operator_str = llm_output[start:i]
                    break
                if c == "(":
                    open_parans += 1
                elif c == ")":
                    close += 1
                i+=1
            
            # Add in the PDDLGym syntax for operator vs. action
            operator_str, debug_msg = self._check_and_fix_operator_string(types, operator_str)
            if operator_str:
                action_pddls.append(operator_str)
            else:
                print(debug_msg)

            end = match.start()


        header = self._create_header()
        output = header + "\n".join(action_pddls)

        print("Parsed:\n", output)

        domain_fname = tempfile.NamedTemporaryFile(delete=False).name
        with open(domain_fname, "w", encoding="utf-8") as f:
            f.write(output)
        domain = PDDLDomainParser(domain_fname)
        return list(domain.operators.values())

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
 
