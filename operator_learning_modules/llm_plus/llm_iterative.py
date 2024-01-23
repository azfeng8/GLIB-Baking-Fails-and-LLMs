"""
Strategy: LLM proposes operators based on training data, and score all the learned operators + LLM's operators every once in a while.

This mainly helps when there is some non-no-op data, but not enough for the learner to make a meaningful operator.

Problem: the LLM is not helping when there is no no-op data. Perhaps iteratively do the warm-starting + score preconditions, especially of actions without learned operators.

#GOAL: without hardcode LLM outputs, run for a few hundred iterations, logging the LLM outputs, to get better perf than LNDR. Then hardcode the collected LLM outputs, and reproduce results
#GOAL: try on more domains other than Baking: Rearrangement, Minecraft, Travel, and get better results than LNDR.
#GOAL: try combining with the warm-starting operators.
"""

from ndr.learn import print_rule_set
from operator_learning_modules.llm_plus.prompts import STATE_TRANSLATION_PROMPT
from operator_learning_modules.llm_plus.operator_search import LEAP_operator_search, LEAP_coverage
from operator_learning_modules.llm_plus.llm_parsing import LLM_PDDL_Parser
from operator_learning_modules import ZPKOperatorLearningModule
from openai_interface import OpenAI_Model
from pddlgym.parser import Operator
from pddlgym.structs import Anti, TypedEntity
from settings import AgentConfig as ac

from abc import abstractmethod
from collections import defaultdict
import pickle
import pdb
import os
from typing import Iterable

### Debugging params
LOGGING = True
READING_DATASET = False
READING_LLM_RESPONSES = False
READING_LEARNING_MOD_OPS = False
LOG_PATH_READ = f'/home/catalan/temp/experiment3/iter_600'
LOG_PATH_WRITE = f'/home/catalan/temp/experiment7'

class BaseLLMIterativeOperatorLearningModule:
    """LLM + learning algorithm combination method. Subclass this with the specific learning algorithm.
    """
    def __init__(self, learned_operators, domain_name, llm, llm_learned_ops):
        self._llm:OpenAI_Model = llm
        self._llm_learn_interval = ac.LLM_learn_interval[domain_name]
        self._learned_operators = learned_operators
        self._llm_learned_ops = llm_learned_ops

        observation_predicates = {p.name: p for p in ac.train_env.observation_space.predicates}
        action_predicates = {p.name: p for p in ac.train_env.action_space.predicates}
        types = set()
        for p in [l for l in ac.train_env.observation_space.predicates] + [a for a in ac.train_env.action_space.predicates]:
            for v in p.var_types:
                types.add(v)
        self._llm_pddl_parser = LLM_PDDL_Parser(action_predicates, observation_predicates, types)

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

        # if itr % self._llm_learn_interval != 0:
            # Debugging. Check that the learner operators are updated correctly.
            # print("\n\nLEARNER NDRS, AFTER UPDATING\n")
            # print_rule_set(self.learner._ndrs)
            # for action_pred in self.learner._transitions:
            #     if "cleanpan" in  action_pred.name:
            #         print("Dataset", self.learner._transitions[action_pred])

        if itr % self._llm_learn_interval != 0:
            return is_updated

        if READING_DATASET:
            with open(f'{LOG_PATH_READ}/trajectories.pkl', 'rb') as f:
                self._trajectories = pickle.load(f)
                for e in self._trajectories:
                    for t in e:
                        self.learner._transitions[t[1].predicate].append((t[0].literals, t[1], t[2]))

        if READING_LEARNING_MOD_OPS:
            with open(f'{LOG_PATH_READ}/learner_ops.pkl', 'rb') as f:
                lops = pickle.load(f)
                self.learner._learned_operators.clear()
                for o in lops:
                    self.learner._learned_operators.add(o)

        # sample random episode. Use all the actions in the episode.
        traj = self._sample_trajectory()

        # No data to sample from.
        if traj is None:
            return is_updated

        if READING_LLM_RESPONSES:
            with open(os.path.join(LOG_PATH_READ, "traj.pkl"), 'rb') as f:
                traj = pickle.load(f)
        if LOGGING:
            os.makedirs(f'{LOG_PATH_WRITE}/iter_{itr}', exist_ok=True)
            with open(f"{LOG_PATH_WRITE}/iter_{itr}/traj.pkl", 'wb') as f:
                pickle.dump(traj, f)
            with open(f'{LOG_PATH_WRITE}/iter_{itr}/learner_ops.pkl', 'wb') as f:
                pickle.dump(self.learner._learned_operators, f)
            with open(f'{LOG_PATH_WRITE}/iter_{itr}/trajectories.pkl', 'wb') as f:
                pickle.dump(self._trajectories, f)

        # LLM proposes new operator for each of the actions in the trajectory.

        llm_ops = self._propose_operators(traj, itr)


        print('\n\nFROM LLM\n')
        for o in llm_ops:
            print(o)
        print('\n\nFROM LEARNER\n')
        for o in self.learner._learned_operators:
            print(o)

        if LOGGING:
            with open(f'{LOG_PATH_WRITE}/iter_{itr}/llm_proposed_ops.pkl', 'wb') as f:
                pickle.dump(llm_ops, f)
        # score and filter the PDDL operators
        ops = self._score_and_filter(llm_ops, itr)

        # update learner operators
        #TODO: remove?
        is_updated =  self._update_operator_rep(ops, itr)

        if LOGGING:
            with open(f'{LOG_PATH_WRITE}/iter_{itr}/ndrs.pkl', 'wb') as f:
                pickle.dump(self._ndrs, f)
            with open(f'/home/catalan/temp/iter_{itr}/updated_learned_ops.pkl', 'wb') as f:
                pickle.dump(self._learned_operators, f)

        return is_updated
        
            
    def _update_state_traj(self, window:tuple):
        """Return the trajectory of the given window, and update state variables.

        Args:
            window (tuple): episode_index, start_index, end_index

        Returns:
            [ transition ]: each transition is tuple of (state literals [frozenset or set], action literal, effects literals [set])
        """
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
        candidates = []
        for ep_i, j in uncovered_transition_indices:
            _, action, _ = self._trajectories[ep_i][j]
            llm_seen_action =  (action.predicate.name.rstrip('0123456789') in self._llm_proposed_actions)
            for window in get_windows(ep_i, j):
                if window not in self._llm_proposed_traj:
                    if llm_seen_action:
                        candidates.append(window)
                    else:
                        # This is the ideal candidate window. Include one per uncovered transition, and choose randomly.
                        return self._update_state_traj(window)

        if len(candidates) > 0:
            window = candidates[np.random.choice(len(candidates))]
            return self._update_state_traj(window)
                
        ### 2.
        # Look through the learner's dataset to find a transition with action LLM hasn't seen yet
        # If found some, then iterate through the dataset in search for a transition with those action
        #  The first one not seen by LLM gets picked.
        ### 3.
        # When iterating through dataset in 2. , add the window for consideration if LLM has seen the actions but not the trajectory yet.
        found_unseen_action = False
        for action_pred in self.learner._transitions:
            if action_pred not in self._llm_proposed_actions:
                found_unseen_action = True
                break
        for episode_i in range(len(self._trajectories)):
            for transition_i in range(len(self._trajectories[episode_i])):
                action = self._trajectories[episode_i][transition_i][1]
                if found_unseen_action and action.predicate == action_pred:
                    for window in get_windows(episode_i, transition_i):
                        return self._update_state_traj(window)
                else:
                    for window in get_windows(episode_i, transition_i):
                        if window not in self._llm_proposed_traj:
                            candidates.append(window)

        if len(candidates) > 0:
            window = candidates[np.random.choice(len(candidates))]
            return self._update_state_traj(window)
        
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
            with open(f'{LOG_PATH_READ}/response_files.pkl', 'rb') as f:
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
                    with open(f'{LOG_PATH_WRITE}/iter_{itr}/response_files.pkl', 'wb') as f:
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
        objects = set()
        for pred in (preds + [p for p in env.action_space.predicates]):
            for v_type in pred.var_types:
                objects.add(v_type)
        object_types = '\n'.join(objects)
        operators = []
        for conversation in op_convos:
            prompt = f"""Given these predicates and object types, translate the description into a PDDL operator:
            Predicates:
            {predicates}

            Object types:
            {object_types}

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
                    with open(f'{LOG_PATH_WRITE}/iter_{itr}/response_files.pkl', 'wb') as f:
                        pickle.dump(response_paths, f)

            op = self._llm_pddl_parser.parse_operator(response)
            if op is not None:
                operators.append(op)

        if LOGGING:
            with open(f'{LOG_PATH_WRITE}/iter_{itr}/response_files.pkl', 'wb') as f:
                pickle.dump(response_paths, f)
        return operators

    def _score_and_filter(self, llm_ops:list[Operator], iter) -> list[Operator]:
        """Score all the operators (from LLM and learner) and return the best operator set.

        Keeps track of which LLM proposed operators were newly added.

        Args:
            llm_ops (list[Operator]): operators proposed by LLM
            iter (int): iteration number.

        Returns:
            list[Operator]: _description_
        """
        llm_end_index = len(llm_ops)
        all_ops = llm_ops + list(self.learner._learned_operators)

        # Renumber the operators so names don't clash.
        # NOTE: Assume initially, operator names are action names or have digits at the end of them.
        # Strip the trailing digits.
        for op in all_ops:
            op.name = op.name.rstrip('0123456789')
        unique_names = defaultdict(lambda: 0)
        for op in all_ops:
            i = unique_names[op.name]
            unique_names[op.name] += 1
            op.name = f"{op.name}{i}"

        op_idxes = LEAP_operator_search(all_ops, self._trajectories, iter)
        llm_accepted_op_i = []
        for op_i in op_idxes:
            if op_i < llm_end_index:
                llm_accepted_op_i.append(op_i)
        learner_ops_same_action = []
        for op_i in llm_accepted_op_i:
            action_name = all_ops[op_i].name.rstrip('0123456789')
            candidates = []
            for learner_op_i in range(llm_end_index, len(all_ops)):
                if action_name == all_ops[learner_op_i].name.rstrip('012345689'):
                    candidates.append(learner_op_i)
            if len(candidates) == 0:
                learner_ops_same_action.append(None)
            else:
                learner_ops_same_action.append(np.random.choice(candidates))

        arg = []
        for  j in learner_ops_same_action:
            if j is None:
                arg.append(None)
            else:
                arg.append(all_ops[j])
        self._update_llm_learned_ops([all_ops[i] for i in llm_accepted_op_i], arg)

        ops = [all_ops[i] for i in op_idxes]
        return ops
   
    def _update_llm_learned_ops(self, llm_accepted_ops, learner_ops_same_action):
        """Update the set of operators proposed by LLM that are good.

        Don't just add to the previous set because the learner may have a better operator
        by now: clear the previous set.
        
        The LLM proposes operators at intervals large enough for several episodes to happen and
        give the exploration a chance to actually collect the needed data.
        
        Args:
            llm_accepted_ops: operators from LLM after scoring/filtering.
            learner_ops_same_action: operators from the learner before scoring/filtering with same action predicate as the corresponding one in the `llm_accepted_ops` list.
        """
        self._llm_learned_ops.clear()
        for o,lo in zip(llm_accepted_ops, learner_ops_same_action):
            self._llm_learned_ops[o] = lo
        print("\n\nUPDATED from LLM ITERATIVE\n")
        print(self._llm_learned_ops)

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
    def __init__(self, learned_operators, domain_name, llm, llm_learned_operators):
        self.learner = ZPKOperatorLearningModule(learned_operators, domain_name)
        self._ndrs = self.learner._ndrs

        super().__init__(learned_operators, domain_name, llm, llm_learned_operators)

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

        print("\n\nUPDATED\n")
        for o in self._learned_operators:
            print(o)

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