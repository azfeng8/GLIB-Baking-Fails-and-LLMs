#!/usr/bin/env python3

from curiosity_modules.curiosity_base import BaseCuriosityModule
from curiosity_modules.goal_babbling import GoalBabblingCuriosityModule
from settings import AgentConfig as ac
from pddlgym import structs
from pddlgym.inference import find_satisfying_assignments

from collections import defaultdict
import copy
import itertools
import numpy as np
import openai
import os
import re
from typing import List


# class LLMActions(BaseCuriosityModule):
#     """Selects actions based on current state and goal.
#     """

# class LLM_GLIB(BaseCuriosityModule):
#     """Babbles goals to achieve.
#     """

class LLMOracle(BaseCuriosityModule):
    """Selects actions by:
    If any action predicate defaults to default rule, take that action.
    Query transition model what action has inaccurate effects and executes that action.
    """
    ### Initialization ###

    def _initialize(self):
        super()._initialize()
        self._rand_state = np.random.RandomState(seed=ac.seed)
        self._name = "llm-oracle"
        self._episode_start_state = None
        self._client = client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    def _get_domain_pddl(self):
        return self._planning_module.create_dom_str()

    def _get_completion(self, msgs):
    
        chat_completion = self._client.chat.completions.create(
            messages=msgs,
            
            model="gpt-4",
        )
        return chat_completion.choices[0].message.content

    ### Reset ###

    def _iw_reset(self):
        pass

    def reset_episode(self, state):
        super().reset_episode(state)
        self._episode_start_state = state

    def learning_callback(self):
        super().learning_callback()
        self._iw_reset()

    ### Get an action ###
    def get_action(self, state, iter_path=None):
        # TODO: If the action does not have a decision tree in the learned model, take that action instead of using LLM.
        
        prompts = ["""You are an reinforcement learning agent in a Baking domain, and you are learning a relational PDDL transition model. Your job is to identify one PDDL operator that has wrong effects in the relational transition model to improve the model.""",
                   "Your next task is to select one of the operators with wrong effects that is most likely to be executed in your current state:.",
                   f"""The PDDL model that you have learned is:\n{self._get_domain_pddl()}""",
                   "Your current state is:\n" + '\n'.join([f"{l.predicate.name}(" + ",".join(l.pddl_variables()) + ")" for l in list(state.literals)]),
                   ]
        msgs = []
        for prompt in prompts:
            msgs.append({ "role": "user", "content": prompt, })
        print("Sending request")
        response = self._get_completion(msgs)
        print("Got response", response)
        action = self._parse_action(response, state)
        action = self._ground_action(action,state)
        print(f"Current state:\n" + '\n'.join([f"{l.predicate.name}(" + ",".join(l.pddl_variables()) + ")" for l in list(state.literals)]), "\nTook action", action)
        return action

    def observe(self, state, action, _effects):
        pass

    def _ground_action(self, action, state):
        # set of possible objects for each argument of the action
        arg_sets:List[set] = []
        objects = state.objects

        for v_type in action.var_types:
            s = set()
            for o in objects:
                if o.var_type == v_type:
                    s.add(o)
            arg_sets.append(s)
        
        #TODO:  Ask GPT to ground the action, potentially based on the state.
        # msgs = []
        # self._get_completion(msgs)

        ### For now, just select a random grounding for the action.
        args = []
        for a in arg_sets:
            choices = a - set(args)
            if len(choices) == 0:
                return self._get_fallback_action(state)
            args.append(sorted(choices)[self._rand_state.choice(len(choices))])
        return action(*args)

    def _get_fallback_action(self, state):
        return super()._get_fallback_action(state)

    def _parse_action(self, response, state):
        # See which predicate appears the most in the response, and select that one.
        ops = {operator.name:operator for operator in self._action_space.predicates}
        names = list(ops.keys())
        counts = []
        for name in names:
            counts.append(len(re.findall(name, response)))
        action = ops[names[np.argmax(counts)]]

        return action