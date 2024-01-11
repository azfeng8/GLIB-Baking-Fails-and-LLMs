import pickle
import re
from collections import defaultdict
from operator_learning_modules import ZPKOperatorLearningModule
from settings import AgentConfig as ac
from openai_interface import OpenAI_Model
from pddlgym.parser import Operator
from pddlgym.structs import Anti, Type, LiteralConjunction, Literal,TypedEntity
from abc import abstractmethod

def _score(operators):
    """Score a set of operators."""
    raise NotImplementedError("TODO")

def _operator_search(operators, score_fn):
    """Search over operator sets for best score."""
    raise NotImplementedError("TODO")

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

        # List of of (s,a,effects) in the current episode
        self._trajectory = []
        
    def observe(self, state, action, effects, start_episode=False, **kwargs):
        """Observe a transition.

        Args:
            state (_type_): _description_
            action (_type_): _description_
            effects (_type_): _description_
            start_episode (bool, optional): _description_. Defaults to False.
        """

        self.learner.observe(state, action, effects)

        # TODO: When episode ends, discard trajectories of length 0.
        if start_episode:
            self._trajectory = []

        # exclude no-ops
        if len(effects) != 0:
            self._trajectory.append((state, action, effects))
        # else:
        #     print('noop')


    def learn(self):
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

        #TODO: if no trajectories yet, log fail message.
        
        # LLM proposes new operator for each of the actions in the trajectory.
        ops = self._propose_operators(traj)

        # score and filter the PDDL operators
        ops.extend(self.learner._learned_operators)
        ops = self._score_and_filter(ops)

        # update learner operators
        is_updated =  self._update_operator_rep(ops)

        self._learn_iter += 1
        return is_updated
        
            
    def _sample_trajectory(self):
        """Returns the current trajectory for the episode.

        Later can cap the number of actions in trajectory and sample from a random episode. 
        """
        assert len(self._trajectory) > 0

        init =  self._trajectory[0]
        init_state, _, _ = init

        # Try to get longest trajectory with different goal state than initial state
        for i,t in enumerate(self._trajectory[::-1]):
            end_state, _, _ = t
            if init_state.literals != end_state.literals:
                return self._trajectory[:-i]
        
        return self._trajectory

    def _propose_operators(self, transitions):
        """
        Args:
            transitions (list): list of (s,a,effects) transitions in the trajectory
        """
        #TODO: change the few-shots to examples from a different domain.
        prompt_few_shot = """;;;; Translate the following state into natural language.

Q:
isflour(flour-0:ingredient), panisclean(pan-0:pan), isflour(flour-1:ingredient), hypothetical(new-1:ingredient), isegg(egg-1:ingredient), hypothetical(new-0:ingredient), isegg(egg-0:ingredient)

A:
There is flour (flour-0) as an ingredient, and the pan (pan-0) is clean. Additionally, there is another type of flour (flour-1) as an ingredient, and a hypothetical new ingredient (new-1). Furthermore, there is an egg (egg-1) as an ingredient, and another hypothetical new ingredient (new-0). Finally, there is an egg (egg-0) as another ingredient.

Q:
isflour(flour-0:ingredient), paninoven(pan-0:pan), panisclean(pan-0:pan), isflour(flour-1:ingredient), ovenisfull(oven-0:oven), inoven(pan-0:pan,oven-0:oven), hypothetical(new-1:ingredient), isegg(egg-1:ingredient), hypothetical(new-0:ingredient), isegg(egg-0:ingredient)

A:
There is flour (flour-0) as an ingredient, and a pan (pan-0) is in the oven (oven-0) that is also clean. Additionally, there is another type of flour (flour-1) as an ingredient. The oven (oven-0) is full, and the pan (pan-0) is inside the oven. Moreover, there is a hypothetical new ingredient (new-1). An egg (egg-1) is also present as an ingredient. Furthermore, there is another hypothetical new ingredient (new-0), and there is an egg (egg-0) as another ingredient.
"""
        # translate the start state into natural language.
        init_state, _, _ = transitions[0]
        prompt_start = prompt_few_shot + f"\nQ:\n" + " ".join([str(l) for l in init_state.literals]) + "\nA:\n"
        #TODO: May vary the temperature on this one
        # init_state_description = self._llm.sample_completions([{"role": "user", "content": prompt_start}], 0, ac.seed, 1)[0]
        #TODO: Uncomment. hardcode example to avoid spurious queries during dev
        init_state_description = "'The pan (pan-0) is clean. There is flour (flour-0) and an egg (egg-0) as ingredients. Additionally, there is a hypothetical new ingredient (new-0).'"

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
        #TODO: May vary the temperature on this one
        #TODO: Uncomment. hardcode example to avoid spurious queries during dev
        # goal_state_desription = self._llm.sample_completions([{"role": "user", "content": prompt_goal}], 0, ac.seed, 1)[0]
        goal_state_desription = "The pan (pan-0) is clean and the oven (oven-0) is full. There is a hypothetical new ingredient (new-0). The pan (pan-0) is in the oven (oven-0). There is flour (flour-0) as an ingredient and an egg (egg-0) as another ingredient."

        # create the task decomposition 
        task_decomp = []
        task_decomp_str = ""
        for _, a, _ in transitions:
            task_decomp.append(f"{a.predicate}")
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
            #TODO: May vary the temperature on this one
            #TODO: Uncomment. hardcode example to avoid spurious queries during dev
            # response = self._llm.sample_completions([{"role": "user", "content": prompt_operator}], 0, ac.seed, 1)[0]
            response = 'The "putpaninoven" operator in natural language can be described as follows:\n\nThis operator represents the action of placing a pan into an oven. In this specific context, the pan is identified as \'pan-0\' and the oven as \'oven-0\'. The initial state before this action is performed is that the pan is clean and outside the oven. There are also ingredients present, namely flour, an egg, and a hypothetical new ingredient. \n\nWhen the "putpaninoven" action is performed, the pan is moved from its initial location and placed inside the oven. The state of the pan remains clean and the oven is now considered full. The ingredients remain the same, with the flour, egg, and hypothetical new ingredient still present. The goal state is achieved when the pan is successfully placed inside the oven.'
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
            #TODO: Uncomment. hardcode example to avoid spurious queries during dev
            # response = self._llm.sample_completions(conversation, 0, ac.seed, 1)[0]
            response = """Sure, here is the PDDL operator for the action "putpaninoven":

```pddl
(:action putpaninoven
    :parameters (?v0 - pan ?v1 - oven)
    :precondition (and 
                    (panisclean ?v0)
                    (not (ovenisfull ?v1))
                  )
    :effect (and 
              (paninoven ?v0)
              (ovenisfull ?v1)
              (not (panisclean ?v0))
            )
)
```

This operator represents the action of putting a clean pan in an oven that is not full. After the action, the pan is in the oven, the oven is full, and the pan is no longer clean."""
            operators.append(self._parse_operator(response))

        with open("/home/catalan/temp/ops.pkl", 'wb') as f:
            pickle.dump(operators, f)

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

        return Operator(op_name, params, preconds, effects)


    def _score_and_filter(self, ops:list[Operator]) -> list[Operator]:

        ops = self._renumber_operators(ops)
        ops = _operator_search(ops, _score)
        return ops
    
    def _renumber_operators(self, ops:list[Operator]) -> list[Operator]:
        """Rename the operators so names are all different.
        """
        # NOTE: Assume initially, operator names are action names or have digits at the end of them.
        # Strip the trailing digits.
        for op in ops:
            op.name = op.name.rstrip('0123456789')
            print(op.name)
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
    def update_operator_rep(self):
        raise NotImplementedError("Override me!")
        
class LLMZPKIterativeOperatorLearningModule(BaseLLMIterativeOperatorLearningModule):
    def __init__(self, learned_operators, domain_name, llm):
        self.learner = ZPKOperatorLearningModule(learned_operators, domain_name)

        super().__init__(learned_operators, domain_name, llm)

    def _update_operator_rep(self, ops:list[Operator]) -> bool:
        """Update the PDDL operator representation, such as NDRs or FOLDTs.

        Args:
            ops (list[Operator])

        Returns:
            bool: if the operators have changed
        """
        self._ndrs = self.learner._ndrs
        # update NDRs and learned_operators
        raise NotImplementedError("TODO")
        return True