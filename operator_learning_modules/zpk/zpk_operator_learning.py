import numpy as np
from settings import AgentConfig as ac
from pddlgym.parser import PDDLDomainParser
from pddlgym.structs import TypedEntity, ground_literal
from ndr.learn import run_main_search as learn_ndrs
from ndr.learn import get_transition_likelihood, print_rule_set, iter_variable_names
from ndr.ndrs import NOISE_OUTCOME, NDR, NDRSet
import re
import openai
import os

from collections import defaultdict
import tempfile


class ZPKOperatorLearningModule:

    def __init__(self, learned_operators, domain_name):
        self._domain_name = domain_name
        self._learned_operators = learned_operators
        self._transitions = defaultdict(list)
        self._seed = ac.seed
        self._rand_state = np.random.RandomState(seed=ac.seed)
        self._learning_on = True
        self._ndrs = {}
        self._fits_all_data = defaultdict(bool)

    def observe(self, state, action, effects):
        if not self._learning_on:
            return
        self._transitions[action.predicate].append((state.literals, action, effects))

        # Check whether we'll need to relearn
        if self._fits_all_data[action.predicate]:
            ndr = self._ndrs[action.predicate]
            if not self._ndr_fits_data(ndr, state, action, effects):
                self._fits_all_data[action.predicate] = False

    def learn(self):
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
            self._learned_operators.clear()
            for ndr_set in self._ndrs.values():
                for i, ndr in enumerate(ndr_set):
                    operator = ndr.determinize(name_suffix=i)
                    # No point in adding an empty effect or noisy effect operator
                    if len(operator.effects.literals) == 0 or NOISE_OUTCOME in operator.effects.literals:
                        continue
                    self._learned_operators.add(operator)

            print_rule_set(self._ndrs)

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

    def _ndr_fits_data(self, ndr, state, action, effects):
        prediction = ndr.predict_max(state.literals, action)
        return sorted(prediction) == sorted(effects)
        # return abs(1 - self.get_probability((state.literals, action, effects))) < 1e-5


class LLMZPKOperatorLearningModule(ZPKOperatorLearningModule):
    """The ZPK operator learner but initialized with operators output by an LLM."""

    def __init__(self, learned_operators, domain_name):
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

        self._openai = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
        actions = '; (:actions ' + ' '.join([p.name for p in env.action_space.predicates]) + ")"
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
        # TODO: cache and make settings
        # reference: https://github.com/Learning-and-Intelligent-Systems/llm4pddl/blob/main/llm4pddl/llm_interface.py

        # TODO: uncomment. Leaving commented for now to avoid spurious queries
        # of the expensive open AI API. Also we might want to use ChatGPT instead...
        completion = self._openai.chat.completions.create(
            model="gpt-4",
            messages = [{"role":"user", "content": prompt}],
            max_tokens=4096,
            temperature=0,
        )
        response = completion.choices[0].text
        print("Got response", response)
        return response
    
    def _check_operator_string(self, types, operator_str):
        """
        Parses the PDDL operator from the string and returns False if the operator string is malformed.

        Args:
            types (set[str]): set of object types allowed in this domain
            operator_str (str): string containing a PDDL operator definition
        Returns:
            bool, str: returns False, error_msg if the string is malformed.
            
        """
        action_name = re.search("[\w]+", operator_str).group(0)
        action_preds = {p.name:p for p in ac.train_env.action_space.predicates}
        if action_name not in action_preds:
            return False,f"Invalid action found: {action_name}"
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
                return False, f"type not found: {var_type} in parameters: {parameter_str}"
            # assert var_type in types, f"type not found: {var_type} in parameters: {parameter_str}"

        # Check each precondition and effect have 1. valid predicate name, 2.predicate argument names match params, 3. predicate argument types match params 
        obs_preds = {o.name:o for o in ac.train_env.observation_space.predicates}
        precond_str = re.search(":precondition([\s\S]*?):effect", operator_str).group(0)[:-7] # trim the ":effect" at the end
        effect_str = re.search(":effect([\s\S]*?)\s\)", operator_str).group(0)
        for pre_or_eff in [precond_str, effect_str]:
            for predicate_str in re.findall("\([\w]+[\s\?\w]*", pre_or_eff):
                if "and" in predicate_str or "not" in predicate_str: continue
                pred_name = re.search("[\w]+", predicate_str).group(0)
                if pred_name != action_name: # extra predicate in preconditions to differ operators vs. actions in PDDLGym
                    if pred_name not in obs_preds:
                        return False, f"Predicate name not valid: {pred_name}, in precondition {pre_or_eff}"
                    # assert pred_name in obs_preds, f"Predicate name not valid: {pred_name}, in precondition {pre_or_eff}"
                arg_types = []
                for arg_name in re.findall("\?[\w\d]+", predicate_str):
                    arg_name = arg_name[1:] # remove the "?" in front
                    if arg_name not in param_names:
                        return False,f"Argument for {pred_name} in {pre_or_eff} not in parameters: {arg_name}"
                    # assert arg_name in param_names, f"Argument for {pred_name} in {pre_or_eff} not in parameters: {arg_name}"
                    arg_types.append(param_types[param_names.index(arg_name)])
                if pred_name == action_name:
                    if arg_types != action_preds[pred_name].var_types:
                        return False, f"Types don't match for predicate {pred_name}: {arg_types} vs. {action_preds[pred_name].var_types}"
                    # assert arg_types == action_preds[pred_name].var_types, f"Types don't match for predicate {pred_name}: {arg_types} vs. {action_preds[pred_name].var_types}"
                else:
                    if arg_types != obs_preds[pred_name].var_types:
                        return False,f"Types don't match for predicate {pred_name} in action {action_name}: {arg_types} vs. {obs_preds[pred_name].var_types}"
                    # assert arg_types == obs_preds[pred_name].var_types, f"Types don't match for predicate {pred_name} in action {action_name}: {arg_types} vs. {obs_preds[pred_name].var_types}"
        return True, ""

    def _llm_output_to_operators(self, llm_output):
        # Parse the LLM output using PDDLGym.
        
        # Automatically detect malformed LLM output.
        # To correct LLM output, currently drop the predicates whose types do not match.

        # Split the response into chunks separated by "(:action ", and discard the first chunk with the header and LLM chat.
        operator_matches = list(re.finditer("\(\:action\s", llm_output))

        action_pddls = []

        # Collect the object types in this domain.
        types = set()
        for p in ac.train_env.action_space.predicates:
            for t in p.var_types:
                types.add(t)

        end = len(llm_output)
        # Check each action's preconditions, parameters, and effects for valid names and consistent typing.
        for match in operator_matches[::-1]: # Read matches from end of file to top of file
            start = match.end()
            operator_str = llm_output[start:end]
            success, debug_msg = self._check_operator_string(types, operator_str)
            if success:
                # Count parantheses: look for the closing to "(:action" to get the operator string.
                open = 0
                close = 0
                i = start
                for c in operator_str:
                    if close > open:
                        action_pddls.append("(:action " + llm_output[start:i])
                        break
                    if c == "(":
                        open += 1
                    elif c == ")":
                        close += 1
                    i+=1
            
            end = match.start()


        header = self._create_header()
        llm_output = header + "\n".join(action_pddls)

        domain_fname = tempfile.NamedTemporaryFile(delete=False).name
        with open(domain_fname, "w", encoding="utf-8") as f:
            f.write(llm_output)
        domain = PDDLDomainParser(domain_fname)
        return list(domain.operators.values())
