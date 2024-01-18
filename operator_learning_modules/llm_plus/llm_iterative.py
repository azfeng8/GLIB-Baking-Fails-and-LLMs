#TODO: finish the TODOs on this page, and then run it from logged outputs.
"""
Strategy: LLM proposes operators based on training data, and score all the learned operators + LLM's operators every once in a while.

Problem: data consists only of successfully executed actions, and LLM only sees this data to improve on the seen operators, not helping explore.

#GOAL: without hardcode LLM outputs, run for a few hundred iterations, logging the LLM outputs, to get better perf than LNDR. Then hardcode the collected LLM outputs, and reproduce results
#GOAL: try on more domains other than Baking: Rearrangement, Minecraft, Travel, and get better results than LNDR.
#GOAL: try combining with the warm-starting operators.
"""

from operator_learning_modules.llm_plus.prompts import STATE_TRANSLATION_PROMPT
from operator_learning_modules.llm_plus.operator_search import LEAP_operator_search, LEAP_coverage
import queue
from typing import Iterable
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

### Debugging params
LOGGING = False
READING_DATASET = True
READING_LLM_RESPONSES = False
READING_LEARNING_MOD_OPS = True
DEBUG_ITER = 900
LOG_PATH = f'/home/catalan/temp/iter_{DEBUG_ITER}'

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
        # Keep track of (episode_index, start_index, end_index) trajectories the LLM has seen, to not propose redundant
        self._llm_proposed_traj = set()
        # Keep track of actions that the LLM has seen, to prioritize unseen actions for the LLM to propose operators for
        self._llm_proposed_actions = set()
        
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
        """Every so often, LLM chooses a trajectory of actions to propose operators, and suggests these along with the learner's
        operators to be considered. An operator search is performed to pick the best operators out of the LLM and learning module's suggested operators.

        Returns:
            bool: if operators were updated
        """
        is_updated = self.learner.learn()
        self._learned_operators = self.learner._learned_operators
        if self._learn_iter % self._llm_learn_interval != 0:
            self._learn_iter += 1
            return is_updated

        if READING_DATASET:
            with open(f'/home/catalan/temp/iter_900/trajectories_900.pkl', 'rb') as f:
                self._trajectories = pickle.load(f)

        if READING_LEARNING_MOD_OPS:
            with open(f'/home/catalan/temp/iter_900/learner_ops_900.pkl', 'rb') as f:
                lops = pickle.load(f)
                self.learner._learned_operators.clear()
                for o in lops:
                    self.learner._learned_operators.add(o)

        # sample random episode. Use all the actions in the episode.
        traj = self._sample_trajectory()

        # No data to sample from.
        if traj is None:
            return is_updated

        if LOGGING:
            os.makedirs(f'/home/catalan/temp/iter_{itr}', exist_ok=True)
            with open(f"/home/catalan/temp/iter_{itr}/traj.pkl", 'wb') as f:
                pickle.dump(traj, f)

        # LLM proposes new operator for each of the actions in the trajectory.

        ops = self._propose_operators(traj, itr)


        print('from llm', ops)
        print('from learner\n', self.learner._learned_operators)

        if LOGGING:
            with open(f'/home/catalan/temp/iter_{itr}/llm_proposed_ops_{itr}.pkl', 'wb') as f:
                pickle.dump(ops, f)
            with open(f'/home/catalan/temp/iter_{itr}/learner_ops_{itr}.pkl', 'wb') as f:
                pickle.dump(self.learner._learned_operators, f)
            with open(f'/home/catalan/temp/iter_{itr}/trajectories_{itr}.pkl', 'wb') as f:
                pickle.dump(self._trajectories, f)

        # score and filter the PDDL operators
        ops.extend(self.learner._learned_operators)
        ops = self._score_and_filter(ops, itr)

        # update learner operators
        is_updated =  self._update_operator_rep(ops, itr)

        if LOGGING:
            with open(f'/home/catalan/temp/iter_{itr}/ndrs.pkl', 'wb') as f:
                pickle.dump(self._ndrs, f)
            with open(f'/home/catalan/temp/iter_{itr}/updated_learned_ops.pkl', 'wb') as f:
                pickle.dump(self._learned_operators, f)

        self._learn_iter += 1
        return is_updated
        
            
    def _update_state_traj(self, window):
        episode, start, end = window
        self._llm_proposed_traj.add(window)
        traj = self._trajectories[episode][start:end + 1]
        for t in traj:
            self._llm_proposed_actions.add(t[1].predicate)
        return traj

    def _sample_trajectory(self) -> list[tuple] or None:
        """Returns the current trajectory for the episode.

        Prioritize in order:
        1. Trajectories including actions that the learner hasn't created operators for yet.
        2. Trajectories with actions that the LLM hasn't seen yet.
        3. Trajectories that have not been proposed to the LLM yet.
        4. All trajectories.

        Returns:
            traj (list[transition]) or None if no data.

        """
        def get_windows(i,j) -> Iterable[tuple]:
            """Get the windows within each episode where:

                - the length of the trajectory is at most ac.max_traj_len
                - the window includes the transition dataset[i][j]
                - the initial state is different from the goal state

            Args:
                i (int): episode index
                j (int): index of transition within episode

            Returns:
                tuple(a,b,c): episode index, start index, end index: the trajectory is dataset[a][b] to dataset[a][c]

            #TODO [algorithm] Get the episode with most uncovered transitions within ac.max_traj_len
                # Idea: generalize get_windows to take in more than one target transition (within the same episode)
            """
            episode, idx = trim_episode(self._trajectories[i], j)
            #TODO [performance] update self._trajectories with the trimmed episode, and only trim each episode one time.
            max_interval_length = ac.max_traj_len
            episode_len = len(episode)

            # Generate the tuples (start, end) that < ac.max_traj_len and include idx. Note end is the index that is included
            for start_index in range(max(idx - max_interval_length + 1, 0), idx + 1):
                for end_index in range(idx, min(episode_len, max(idx, max_interval_length + start_index - 1) + 1))[::-1]: # return longer windows first
                    # Filter out the tuples that don't have initial_state != goal_state
                    init_state, *_ = episode[start_index]
                    end_state, _, effects = episode[end_index]
                    goal_lits = get_next_state(end_state, effects)
                    if frozenset(init_state.literals) != goal_lits:
                        yield (i, start_index, end_index)

                if (sum(len(l) for l in self._trajectories) == 0): return None
        ### 1. 
        # Get uncovered_transitions, the episode index, index within episode
        *_, uncovered_transition_indices = LEAP_coverage(self._trajectories, self.learner._learned_operators)
        candidates = queue.PriorityQueue()
        for ep_i, j in uncovered_transition_indices:
            state, action, eff = self._trajectories[ep_i][j]
            llm_seen_action =  (action.predicate.name.rstrip('0123456789') not in self._llm_proposed_actions)
            for window in get_windows(ep_i, j):
                if window not in self._llm_proposed_traj:
                    if llm_seen_action:
                        candidates.put((1, window))
                    else:
                        # This is the ideal candidate window.
                        candidates.put((0, window))
                        break

        if candidates.qsize() > 0:
            _, window = candidates.get()
            return self._update_state_traj(window)
                
        # TODO: Run with only 1. and 4. implemented. Then fill in 2,3.

        ### 2.
        # Look through the learner's dataset to find a transition with action LLM hasn't seen yet
        # If found some, then iterate through the dataset in search for a transition with those action
        # get_windows(), check if in self._proposed_traj. The first one not in it gets picked.

        ### 3.
        # When iterating through dataset in 2. , add the window to priority queue with priority 3 if LLM has seen the actions but not the trajectory yet.

        # if candidates.qsize() > 0:
        #     _, window = candidates.get()
        #     return self._update_state_traj(window)

        ### 4. Consider all trajectories.

        # Get a random episode's sequence of actions (excluding no-ops)
        for i in np.random.permutation(range(len(self._trajectories))):
            if len(self._trajectories[i]) > 0:
                episode_index = i
                break
        # Randomly sample a transition in the trajectory, and randomly return a window including that transition.
        idx =  np.random.choice(range(len(self._trajectories[episode_index])))
        gen = get_windows(episode_index, idx)
        window = next(gen)
        return self._update_state_traj(window)

    def _propose_operators(self, transitions, itr=-1):
        """
        Args:
            transitions (list): list of (s,a,effects) transitions in the trajectory
            #TODO [algorithm] May vary the temperature on the 3 natural language translation prompts
        """
        read_index = 0
        if LOGGING:
            response_paths = []
        if READING_LLM_RESPONSES:
            with open('/home/catalan/temp/iter_900/response_files.pkl', 'rb') as f:
                response_paths = pickle.load(f)

        assert len(transitions) > 0

        # translate the start state into natural language.
        init_state, _, _ = transitions[0]
        prompt_few_shot = STATE_TRANSLATION_PROMPT 
        prompt_start = prompt_few_shot + f"\nQ:\n" + " ".join([str(l) for l in init_state.literals]) + "\nA:\n"
        if READING_LLM_RESPONSES:
            with open(response_paths[read_index], 'rb') as f:
                init_state_description = pickle.load(f)[0]
                read_index += 1
        else:
            responses, f = self._llm.sample_completions([{"role": "user", "content": prompt_start}], 0, ac.seed, 1)
            init_state_description = responses[0]
            if LOGGING:
                response_paths.append(f)

        # translate the end state into natural language.
        goal_state, _, effects = transitions[-1]

        goal_lits = get_next_state(goal_state, effects)
        prompt_goal = prompt_few_shot + f"\nQ:\n" + " ".join([str(l) for l in goal_lits]) + "\nA:\n"
        if READING_LLM_RESPONSES:
            with open(response_paths[read_index], 'rb') as f:
                goal_state_desription = pickle.load(f)[0]
                read_index += 1
        else:
            responses,f = self._llm.sample_completions([{"role": "user", "content": prompt_goal}], 0, ac.seed, 1)
            goal_state_desription = responses[0]
            if LOGGING:
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
            if READING_LLM_RESPONSES:
                with open(response_paths[read_index], 'rb') as f:
                    response = pickle.load(f)[0]
                read_index += 1
            else:
                responses,f = self._llm.sample_completions([{"role": "user", "content": prompt_operator}], 0, ac.seed, 1)
                response = responses[0]
                if LOGGING:
                    response_paths.append(f)
                    with open(f'/home/catalan/temp/iter_{itr}/response_files.pkl', 'wb') as f:
                        pickle.dump(response_paths, f)

            op_convos.append([{"role": "user", "content": prompt_operator}, {"role": "assistant", "content": response}])

        # Get predicates
        env = ac.train_env
        preds = [p for p in env.observation_space.predicates]
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
            if READING_LLM_RESPONSES:
                with open(response_paths[read_index], 'rb') as f:
                    response = pickle.load(f)[0]
                read_index += 1
            else:
 
                responses,f = self._llm.sample_completions(conversation, 0, ac.seed, 1)
                response = responses[0]
                if LOGGING:
                    response_paths.append(f)
                    with open(f'/home/catalan/temp/iter_{itr}/response_files.pkl', 'wb') as f:
                        pickle.dump(response_paths, f)

            op = self._parse_operator(response)
            if op is not None:
                operators.append(op)

        if LOGGING:
            with open(f'/home/catalan/temp/iter_{itr}/response_files.pkl', 'wb') as f:
                pickle.dump(response_paths, f)
        return operators

    def _parse_operator(self, llm_response:str) -> Operator or None:
        """Parse an Operator from the LLM response.

        Args:
            llm_response (str)

        Raises:
            Exception: Used in debugging only, will remove #TODO:.

        Returns:
            Operator or None: operator that was parsed, or None if not able to parse a non-null-effect operator.
        """
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
        literals = self._get_literals(precond_str, param_names, param_types)

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
        eliterals = self._get_literals(effect_str, param_names, param_types)
        if len(eliterals) == 0: return None
        effects = LiteralConjunction(eliterals)

        # Rename the variables
        var_name_gen = iter_variable_names()
        variables = {}
        for l in effects.literals + preconds.literals:
            for v in l.variables:
                if v not in variables:
                    v_name = next(var_name_gen)
                    variables[v] = Type(v_name)

        literals = []
        for l in preconds.literals:
            args = []
            for v in l.variables:
                args.append(variables[v])
            literals.append(Literal(l.predicate, args))
        preconds = LiteralConjunction(literals)

        literals = []
        for l in effects.literals:
            args = []
            for v in l.variables:
                args.append(variables[v])
            literals.append(Literal(l.predicate, args))
        effects = LiteralConjunction(literals)       

        params = set()
        for l in effects.literals + preconds.literals:
            for v in l.variables:
                params.add(v)
                
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
    
    def _get_literals(self, precond_or_eff_str:str, param_names:list[str], param_types:list[str]) -> list[Literal]:
        """Helper for _parse_operator. Returns Literals in the precondition or effect string.

        A parsed predicate is dropped if:
            - If the parsed predicate argument doesn't appear in the parsed parameters
            - If parsed argument types don't match the predicate argument types

        Returns:
            list[Literal]
        """
        literals = []
        for predicate_str_match in re.finditer("\([\w]+[\s\?\w-]*\)", precond_or_eff_str):
            predicate_str = predicate_str_match.group(0)
            if "and" in predicate_str or "not" in predicate_str: 
                continue
            pred_name = re.search("[\w]+", predicate_str).group(0)

            if pred_name not in self._observation_predicates:
                continue

            args = []
            arg_types = []
            drop_pred = False
            for arg_name in re.findall("\?[\w\d]+", predicate_str):
                arg_name = arg_name[1:] # remove the "?" in front
                if arg_name not in param_names:
                    drop_pred = True
                    break
                arg_type = param_types[param_names.index(arg_name)]
                arg_types.append(arg_type)
                args.append(Type(f"{arg_name}"))

            if drop_pred:
                continue
            predicate = self._observation_predicates[pred_name]
            if predicate.var_types != arg_types:
                continue

            literals.append(Literal(predicate, args))

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

def get_next_state(goal_state, effects):
    goal_lits = set()
    goal_lits |= goal_state.literals
    for e in effects:
        if e.is_anti:
            goal_lits.remove(Anti(e))
        else:
            goal_lits.add(e)
    return frozenset(goal_lits)

def trim_episode(episode:list[tuple], idx:int) -> tuple[list[tuple], int]:
    """Trims out repeated action sequences that don't give more information.

    Keeps the jth transition in the trimmed episode. The new index referring to that transition is returned.

    - Cut out repeated sequences. If the same state reappears more than twice in the sequence, then there are several 'undo' sequences. If the undo sequences repeat, keep only one undo sequence.

    For example, saw in Baking domain: putpaninoven, removepanfromoven, putpaninoven, removepanfromoven ... . Only useful to keep one pair.

    Returns:
        trimmed episode and the index of the target transition.
    """

    # State to Indices of the transitions whose starts are the same state.
    state_occurence_indices = defaultdict(list)
    for i, tran in enumerate(episode):
            state, action, effects = tran
            state_occurence_indices[frozenset(state.literals)].append(i)

    s, a , e = episode[-1]
    state_occurence_indices[get_next_state(s,e)].append(len(episode))
    to_trim = any(len(state_occurence_indices[state]) > 2 for state in state_occurence_indices)

    while to_trim:
            # Trim the most frequently occuring sequence
            state = max(state_occurence_indices, key=lambda x: len(state_occurence_indices[x]))
            x = state_occurence_indices[state][0]
            # Store the action sequence to be kept. Tuple (action sequence) to tuple (cut [x,y))
            action_seqs = {}
            # cuts to make [x,y)
            cuts = []
            for y in state_occurence_indices[state][1:]:
                    action_seq = tuple([t[1] for t in episode[x:y]])
                    if action_seq in action_seqs:
                            # mark it for cutting if it doesn't contain the target transition
                            if not (idx >= x and idx < y):
                                    cuts.append((x,y))
                            else:
                            # if it contains the target transition, mark the existing action seq for cutting, and keep this copy instead.
                                    cuts.append(action_seqs[action_seq])
                                    action_seqs[action_seq] = (x,y)
                    else:
                            action_seqs[action_seq] = (x,y)
                    x = y
            e = episode
            trimmed_episode = []
            prev_end = 0
            for start,end in cuts:
                    if idx > end:
                            idx -= end - start + 1
                    trimmed_episode += episode[prev_end:start]
                    prev_end = end
            if len(cuts) > 0:
                    trimmed_episode += episode[prev_end:]
                    episode = trimmed_episode
            else:
                    break


            # Update state_occurence_indices and idx
            state_occurence_indices = defaultdict(list)
            for i, tran in enumerate(episode):
                    state, action, effects = tran
                    state_occurence_indices[frozenset(state.literals)].append(i)

            s, a , e = episode[-1]
            state_occurence_indices[get_next_state(s,e)].append(len(episode))
            to_trim = any(len(state_occurence_indices[state]) > 2 for state in state_occurence_indices)
    return episode, idx
