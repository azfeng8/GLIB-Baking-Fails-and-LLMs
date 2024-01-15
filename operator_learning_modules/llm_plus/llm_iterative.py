"""
Strategy: LLM proposes operators based on training data, and score all the learned operators + LLM's operators every once in a while.

Problem: data consists only of successfully executed actions, and LLM only sees this data to improve on the seen operators, not helping explore.

#GOAL: without hardcode LLM outputs, run for a few hundred iterations, logging the LLM outputs, to get better perf than LNDR. Then hardcode the collected LLM outputs, and reproduce results
#GOAL: try on more domains other than Baking: Rearrangement, Minecraft, Travel, and get better results than LNDR.
#GOAL: try combining with the warm-starting operators.
"""

from operator_learning_modules.llm_plus.prompts import STATE_TRANSLATION_PROMPT
from operator_learning_modules.llm_plus.operator_search import LEAP_operator_search
import pickle
import re
from collections import defaultdict
from operator_learning_modules import ZPKOperatorLearningModule
from settings import AgentConfig as ac
from openai_interface import OpenAI_Model
from pddlgym.parser import Operator
from pddlgym.structs import Anti, Type, LiteralConjunction, Literal,TypedEntity
from abc import abstractmethod
import pdb
import os



class BaseLLMIterativeOperatorLearningModule:
    """_summary_
    """
    def __init__(self, learned_operators, domain_name, llm):
        self._llm:OpenAI_Model = llm
        self._llm_learn_interval = ac.LLM_learn_interval[domain_name]
        self._learn_iter = 0
        self._learned_operators = learned_operators
        self._observation_predicates = {p.name: p for p in ac.train_env.observation_space.predicates}
        self._action_predicates = {p.name: p for p in ac.train_env.action_space.predicates}

        # Dataset of transitions
        self._trajectories = []
        
    def observe(self, state, action, effects, start_episode=False, **kwargs):
        """Observe a transition.

        Args:
            start_episode (bool, optional): This is the first observation in the current episode. Defaults to False.
        """

        self.learner.observe(state, action, effects)

        if start_episode:
        # When episode ends, discard trajectories of length 0.
            if len(self._trajectories) > 0 and len(self._trajectories[-1]) == 0:
                pass
            else:
                self._trajectories.append([])

        # exclude no-ops
        if len(effects) != 0:
            self._trajectories[-1].append((state, action, effects))


    def learn(self, itr):
        """_summary_

        Returns:
            bool: if operators were updated
        """
        is_updated = self.learner.learn()
        self._learned_operators = self.learner._learned_operators
        if self._learn_iter % self._llm_learn_interval != 0:
            self._learn_iter += 1
            return is_updated

        # sample random episode. Use all the actions in the episode.
        traj = self._sample_trajectory()

        # No data to sample from.
        if traj is None:
            return is_updated

        os.makedirs(f'/home/catalan/temp/iter_{itr}', exist_ok=True)
        # LLM proposes new operator for each of the actions in the trajectory.
        if itr in [0, 300]:
            with open(f'/home/catalan/temp/iter_{itr}/llm_proposed_ops_{itr}.pkl', 'rb') as f:
                ops = pickle.load(f)
            # self._resume(itr)
    
        else:
            ops = self._propose_operators(traj, itr)

        print('from llm', ops)
        print('from learner\n', self.learner._learned_operators)

        with open(f'/home/catalan/temp/iter_{itr}/llm_proposed_ops_{itr}.pkl', 'wb') as f:
            pickle.dump(ops, f)
        with open(f'/home/catalan/temp/iter_{itr}/learner_ops_{itr}.pkl', 'wb') as f:
            pickle.dump(self.learner._learned_operators, f)
        with open(f'/home/catalan/temp/iter_{itr}/trajectories_{itr}.pkl', 'wb') as f:
            pickle.dump(self._trajectories, f)

        # score and filter the PDDL operators
        ops.extend(self.learner._learned_operators)
        ops = self._score_and_filter(ops, itr)

        # print("after scoring\n", ops)

        # update learner operators
        is_updated =  self._update_operator_rep(ops, itr)

        self._learn_iter += 1
        return is_updated
        
            
    def _sample_trajectory(self):
        """Returns the current trajectory for the episode.

        Later can cap the number of actions in trajectory.

        Returns:
            traj (list[transition]) or None if no data.

        """
        #TODO: never sample the same trajectory twice
        #TODO: LLM should propose operators for the uncovered transitions. Those trajectories should be sampled first.

        # Use the most recent episode if possible. Otherwise sample from some past episode
        found = False
        if len(self._trajectories[-1]) == 0:
            for i in np.random.permutation(range(len(self._trajectories))):
                if len(self._trajectories[i]) > 0:
                    traj = self._trajectories[i]
                    found = True
            if not found:
                return None
        else:
            traj = self._trajectories[-1]

        # Randomly sample the start of the trajectory
        init_i =  np.random.choice(range(len(traj)))
        init_state, _, _ = traj[init_i]

        traj = traj[init_i:]
        if len(traj) == 1: return traj

        # Try to get longest trajectory with different goal state than initial state
        for i,t in enumerate(traj[::-1]):
            end_state, _, _ = t
            if init_state.literals != end_state.literals:
                return traj[:len(traj)-i]
        
        return traj

    def _propose_operators(self, transitions, itr=-1):
        """
        Args:
            transitions (list): list of (s,a,effects) transitions in the trajectory
            #TODO: May vary the temperature on the 3 natural language translation prompts
        """
        response_paths = []
        assert len(transitions) > 0
        with open(f"/home/catalan/temp/iter_{itr}/traj.pkl", 'wb') as f:
            pickle.dump(transitions, f)
        # translate the start state into natural language.
        init_state, _, _ = transitions[0]
        prompt_few_shot = STATE_TRANSLATION_PROMPT 
        prompt_start = prompt_few_shot + f"\nQ:\n" + " ".join([str(l) for l in init_state.literals]) + "\nA:\n"
        responses, f = self._llm.sample_completions([{"role": "user", "content": prompt_start}], 0, ac.seed, 1)
        init_state_description = responses[0]
        response_paths.append(f)

        # translate the end state into natural language.
        goal_state, _, effects = transitions[-1]
        goal_lits = set()
        goal_lits |= goal_state.literals
        for e in effects:
            if e.is_anti:
                goal_lits.remove(Anti(e))
            else:
                goal_lits.add(e)
        prompt_goal = prompt_few_shot + f"\nQ:\n" + " ".join([str(l) for l in goal_lits]) + "\nA:\n"
        responses,f = self._llm.sample_completions([{"role": "user", "content": prompt_goal}], 0, ac.seed, 1)
        goal_state_desription = responses[0]
        response_paths.append(f)

        # create the task decomposition 
        task_decomp = set()
        task_decomp_str = ""
        for _, a, _ in transitions:
            task_decomp.add(f"{a.predicate}")
            task_decomp_str += f"{a.predicate}(" + ",".join(a.pddl_variables()) + ") "

        op_convos = []
        for action in task_decomp:
            prompt_operator = f"""
            ;;;;  Given actions from initial state to goal state describe an operator called "{action}" in natural language.

            Actions:
            {task_decomp_str}
            
            Initial State:
            {init_state_description}
            
            Goal State:
            {goal_state_desription}
            """
            responses,f = self._llm.sample_completions([{"role": "user", "content": prompt_operator}], 0, ac.seed, 1)
            response = responses[0]
            response_paths.append(f)
            with open(f'/home/catalan/temp/iter_{itr}/response_files.pkl', 'wb') as f:
                pickle.dump(response_paths, f)

            op_convos.append([{"role": "user", "content": prompt_operator}, {"role": "assistant", "content": response}])

        # Get predicates
        env = ac.train_env
        preds = [p for p in env.action_space.predicates] + [p for p in env.observation_space.predicates]
        lines = []
        for p in preds:
            s = f"({p.name} " + " ".join(p.pddl_variables()) + ")"
            lines.append(s)
        predicates = '\n'.join(lines)
        operators = []
        for conversation in op_convos:
            prompt = f"""Given these predicates, translate the description into a PDDL operator:
            Predicates:
            {predicates}

            Use the format:
            
            (:action 
                :parameters ()
                :precondition (
                )
                :effect (
                )
            
            """
            conversation.append({"role": "user", "content": prompt})
            responses,f = self._llm.sample_completions(conversation, 0, ac.seed, 1)
            response = responses[0]
            response_paths.append(f)
            # This operator represents the action of putting a clean pan in an oven that is not full. After the action, the pan is in the oven, the oven is full, and the pan is no longer clean."""
            operators.append(self._parse_operator(response))

        with open(f'/home/catalan/temp/iter_{itr}/response_files.pkl', 'wb') as f:
            pickle.dump(response_paths, f)
        return operators

    def _parse_operator(self, llm_response:str) -> Operator:
        # Find the PDDL operator in the response.
        match = re.search("\(\:action", llm_response)
        # Count parantheses: look for the closing to "(:action" to get the operator string.
        open_parans = 0
        close = 0
        i = match.end()
        operator_str = None
        for c in llm_response[match.end():]:
            if c == "(":
                open_parans += 1
            elif c == ")":
                close += 1
            if close > open_parans:
                operator_str = llm_response[match.start():i]
                break
            i+=1

        if operator_str is None: raise Exception(f"Parsing error: {llm_response}")
        # Extract operator name.
        match = re.search("\(\:action\s\w+", operator_str)
        op_name = operator_str[match.start() + len("(:action "):match.end()]

        # Extract parameters.
            # NOTE: Assume parameters are written on one line.
        match = re.search("\:parameters[^\)]*\)", operator_str)
        param_str = operator_str[match.start() + len(":parameters ("): match.end()].rstrip(')')
        param_names:list[str] = []
        param_types:list[str] = []
        for s in param_str.split('?'):
            if s == "": continue
            name, var_type = s.split(' - ')
            name = name.strip()
            var_type = var_type.strip()
            param_names.append(name)
            param_types.append(var_type)

        # Extract preconditions.
        match = re.search(":precondition([\s\S]*?):effect", operator_str)
        precond_str = operator_str[match.start() + len(":precondition (") : match.end() - len(":effect")]
        literals = self._get_literals(precond_str, param_names)

        # NOTE: Prompting the action multiple times will result in different operators.
        action_pred = self._action_predicates[op_name]
        args = []
        for v_type in action_pred.var_types:
            v_name = param_names[param_types.index(str(v_type))]
            args.append(Type(v_name))
        action = action_pred(*args)
        preconds = LiteralConjunction(literals + [action])

        # Extract effects.
        effect_str_match = re.search(":effect([\s\S]*?)\s\)", operator_str)
        effect_str = operator_str[effect_str_match.start():effect_str_match.end()]
        eliterals = self._get_literals(effect_str, param_names)
        effects = LiteralConjunction(eliterals)

        # Make parameters
        params = set()
        for literal in effects.literals + preconds.literals:
            for v in literal.variables:
                params.add(v)

        #TODO: need thorough checking of names, types, variable syntax. Importantly, the lifted predicate argument names (e.g. ?x0) should be consistent in parameters, preconditions, and effects.

        return Operator(op_name, params, preconds, effects)


    def _score_and_filter(self, ops:list[Operator], iter) -> list[Operator]:

        ops = self._renumber_operators(ops)
        ops = LEAP_operator_search(ops, self._trajectories, iter)
        return ops
    
    def _renumber_operators(self, ops:list[Operator]) -> list[Operator]:
        """Rename the operators so names are all different.
        """
        # NOTE: Assume initially, operator names are action names or have digits at the end of them.
        # Strip the trailing digits.
        for op in ops:
            op.name = op.name.rstrip('0123456789')
        # Renumber them.
        unique_names = defaultdict(lambda: 0)
        for op in ops:
            i = unique_names[op.name]
            unique_names[op.name] += 1
            op.name = f"{op.name}{i}"
        return ops
    
    def _get_literals(self, precond_or_eff_str, param_names) -> list[Literal]:
        """Helper for _parse_operator.

        Args:
            precond_or_eff_str (_type_): _description_
            param_names (_type_): _description_

        Raises:
            Exception: _description_

        Returns:
            list[Literal]: _description_
        """
        literals = []
        for predicate_str_match in re.finditer("\([\w]+[\s\?\w]*\)", precond_or_eff_str):
            predicate_str = predicate_str_match.group(0)
            if "and" in predicate_str or "not" in predicate_str: continue
            pred_name = re.search("[\w]+", predicate_str).group(0)

            if pred_name not in self._observation_predicates:
                continue

            args = []
            for arg_name in re.findall("\?[\w\d]+", predicate_str):
                arg_name = arg_name[1:] # remove the "?" in front
                if arg_name not in param_names:
                    raise Exception(f"Argument for {pred_name} in not in parameters: {arg_name}")
                # arg_type = param_types[param_names.index(arg_name)]
                args.append(Type(f"{arg_name}"))
            literals.append(Literal(self._observation_predicates[pred_name], args))
        return literals


    @abstractmethod
    def _update_operator_rep(self, itr):
        raise NotImplementedError("Override me!")
    
    @abstractmethod
    def _resume(self):
        raise NotImplementedError("Override me!")
        
import numpy as np
from ndr.ndrs import NDR, NDRSet, NOISE_OUTCOME
from ndr.learn import iter_variable_names
from pddlgym.structs import TypedEntity, ground_literal

class LLMZPKIterativeOperatorLearningModule(BaseLLMIterativeOperatorLearningModule):
    def __init__(self, learned_operators, domain_name, llm):
        self.learner = ZPKOperatorLearningModule(learned_operators, domain_name)
        self._ndrs = self.learner._ndrs

        super().__init__(learned_operators, domain_name, llm)

    def _update_operator_rep(self, ops:list[Operator], itr=-1) -> bool:
        """Update the NDRs.

        Args:
            ops (list[Operator])

        Returns:
            bool: if the operators have been changed
        """
        ndrs = defaultdict(list)
        for op in ops:
            # In initializing the learner from previous, we assume a
            # standard variable naming scheme.
            # This makes sure that grounded `action` is the same for all ops with that action.
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
            ndrs[action].append(ndr)

        # For non no-op data, update the scored operators. For no-op actions, let ZPK do its thing.
        ndr_sets = self.learner._ndrs
        for action in ndrs:
            ndr_sets[action.predicate] = NDRSet(action, ndrs[action])
        self._learned_operators.clear()
        for o in ops:
            self._learned_operators.add(o)
        self._ndrs = ndr_sets
        self.learner._ndrs = ndr_sets

        print("\nupdated", self._learned_operators)

        with open(f'/home/catalan/temp/iter_{itr}/ndrs.pkl', 'wb') as f:
            pickle.dump(self._ndrs, f)
        with open(f'/home/catalan/temp/iter_{itr}/updated_learned_ops.pkl', 'wb') as f:
            pickle.dump(self._learned_operators, f)

        return True
    def _resume(self, itr):

        with open(f'/home/catalan/temp/iter_{itr}/ndrs.pkl', 'rb') as f:
            self._ndrs = pickle.load(f)
            self.learner._ndrs = self._ndrs 
        with open(f'/home/catalan/temp/iter_{itr}/updated_learned_ops.pkl', 'rb') as f:
            self._learned_operators.clear()
            self.learner._learned_operators.clear()
            for o in pickle.load(f):
                self._learned_operators.add(o)
                self.learner._learned_operators.add(o)
        self._learn_iter = itr
