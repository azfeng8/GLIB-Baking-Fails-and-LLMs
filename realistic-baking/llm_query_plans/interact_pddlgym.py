import numpy as np
import json
import re
import gym
import pddlgym
import argparse
import os
from copy import deepcopy
from pprint import pprint
from pddlgym.structs import Anti, State

from openai_interface import OpenAI_Model

def get_objects_of_type(objects_: frozenset, object_type):
    objs = []
    for o in objects_:
        name, obj_type = o._str.split(':') 
        if obj_type.strip() == object_type:
            objs.append(o)
    return objs


def get_facts(descriptions, state_string):
    true_facts = ""
    # Parsing code to get descriptions from the state literals
    for line in state_string.split('\n'):
        line = line.strip()
        if line == '': continue
        items = line[1:-1].split()
        pred_name = items[0]
        description_string = descriptions["predicates"][pred_name]
        description, arg_order = description_string.split('#')
        description = description.strip()

        argument_order = items[1:]

        fstring_object_order = []
        for arg in arg_order.strip():
            fstring_object_order.append(argument_order[int(arg)])
        true_facts += (description.format(*fstring_object_order)) + '\n'
    return true_facts.strip()


class LLMAgent:
    def __init__(self, env):
        """Create the prompting agent.
        """
        self.llm = OpenAI_Model()
        self.env = env
        self.action_preds = {p.name: p for p in env.action_space.predicates}
        self._get_intro()
   
    def _get_intro(self):
        conversation = []
        intro_prompt = \
        f"""
        You are a household robot in a kitchen. You are in front of the kitchen counter, where there are some prepared ingredients. 

        More specifically, you will be given a set of facts that are currently true in the world, and a set of facts that is your goal to make true in the world. With my step-by-step guidance, you will think through how to act to achieve the goal.

        Since you are baking desserts, first determine what are the differences between a cake and sweet, light, and airy souffle. Please rationalize what are the essential ingredients and their amounts to make those desserts and use only those.
        """
        # print("**********************PROMPT********************")
        # print(intro_prompt)
        conversation.append( {"role": "user", "content": intro_prompt})
        responses, _ = self.llm.sample_completions(self.conversation, num_completions=1, temperature=0, seed=1)
        conversation.append({"role": "assistant", 'content': responses[0]})
        self.intro_conversation = conversation


    def get_action(self, obs, problem_idx):
        """Prompt LLM for the next action.

        Args:
            obs (pddlgym.structs.State): observation.
        """
        action = self._query_LLM_for_ground_action(obs, problem_idx)
        return action 

    
    def _query_LLM_for_ground_action(self, obs, problem_idx):
        """Queries LLM and returns the ground action predicate.
        """
        conversation = deepcopy(self.intro_conversation) # Conversation with the LLM for this action
        with open('predicate_and_goal_descriptions.json', 'r') as f:
            descriptions = json.load(f)
        initial_state = ''
        for lit in obs.literals:
            if lit.predicate.name not in ('different', 'name-less-than') and lit.predicate not in self.env.action_space.predicates:
                initial_state += lit.pddl_str() + '\n'
        initial_state_predicate_fstrings = get_facts(descriptions, initial_state)
        problem_name = f'problem{problem_idx+1}'
        goal_state_predicate_fstrings = descriptions['train_goals'][problem_name]
        problem_setting_prompt = \
        f"""
        In the kitchen, there different kinds of objects that you can interact with. The different kind of objects that you see are categorized into the following:

        container
        measuring cup
        dessert
        powder
        butter
        mixture
        egg
        oven
        spatula
        electric stand mixer

        Right now, you see the some of these ingredients and items on the counter. You also see some appliances in the kitchen. The following things are true at this moment:

        {initial_state_predicate_fstrings}

        As a reminder, in the kitchen, the pans, measuring cups, and bowls are on the counter, and the oven(s) is (are) behind the counter. If you are baking desserts, please rationalize what are the essential ingredients and their amounts to make those desserts and use only those. Once an ingredient is used once, it can't be reused.

        You should have all of the ingredients that you need on the counter prepared for you. I'll let you know what desserts you will make shortly. 
        """
        print("**********************PROMPT********************")
        print(problem_setting_prompt)
        conversation.append({"role": "user", "content": problem_setting_prompt})
        responses, _ = self.llm.sample_completions(conversation, num_completions=1, temperature=0, seed=1)
        conversation.append({"role": "assistant", 'content': responses[0]})

        action_description_string = ""
        action_names_string = ""
        for k, v in descriptions["lifted_skill_descriptions"].items():
            action_description_string += k + ": " + v + '\n'
            action_names_string += k  + '\n'
        formalizing_intro = \
        f"""These are the things that you would like to become true:
        {goal_state_predicate_fstrings}

        This state is your goal.

        These are the names of the atomic actions that we can perform, along with their descriptions:
        {action_description_string}

        Can you please give a sequence of these phrases that will get us to the goal? Format it using a numbered list with one line per step, starting with "1.". Give a little explanation of each step underneath each bullet point. Mark the end of the plan with '***' in your response.
        """
        print("**********************PROMPT********************")
        print(formalizing_intro)
        input("Insert response") 
        response =  "" #TODO:: get response from API
        match = re.search('\s1\.(.|\n)*\s2\.', response)
        if match is None:
            # Maybe there are fewer than two steps to the goal.
            match = re.search('\s1\.(.|\n)*\*\*\*', response)
            first_instruction = response[match.start():match.end() - len('***')].strip()
        else:
            first_instruction = response[match.start():match.end() - len('2.')]
        assert match is not None, f'Parsing failed for finding step 1 in plan: {response}'
        action_name = None
        for action_pred_name in self.action_preds:
            if action_pred_name in first_instruction:
                action_name = action_pred_name
                break
        assert action_name is not None, f"No action name found in: {first_instruction}"
        variable_description_list = descriptions["skill_variable_descriptions"][action_name]
        grounding_prompt = \
        f"""Thanks. Let's think step by step what objects are associated with each of these actions.
        Let's recap what we've talked about. Currently, the following facts are true:

        {initial_state_predicate_fstrings}

        We want to make these facts true:
        {goal_state_predicate_fstrings}

        We're thinking through a plan step-by-step to our goal. You gave a sketch of the plan in the response above.

        We are going to do the first step in the plan. We need to identify the names of the specific objects involved in this action. Here are more details about how the objects involved need to relate to the action.
        """ + '\n'.join(variable_description_list) 
        print(grounding_prompt)
        input()

        ground_objs = []
        for i, variable_description in enumerate(variable_description_list):
            object_type = self.action_preds[action_name].var_types[i]
            objects_list = get_objects_of_type(obs.objects, object_type)

            if len(objects_list) == 1:
                ground_objs.append(objects_list[0])
            else:
                action_description_with_nonspecific_articles = descriptions["lifted_skill_descriptions"][action_name]
                action_grounding_variable_prompt = \
    f"""We are going to {action_description_with_nonspecific_articles[:-1].lower()}. Given knowledge of the current state and our planned actions, which of the following objects fits the description, {variable_description}?
    """ + '\n'.join(objects_list) + '\n' + 'Please explain your answer, and then answer with the object name on the last line after "Answer:".'

                print(action_grounding_variable_prompt)
                response = input()
                match = re.search("Answer\:\s*\w+", response)
                assert match is not None, response
                obj_name = response[match.start() + len('Answer:'): match.end()]
                ground_objs.append([o for o in objects_list if o._str.split(':')[0] == obj_name][0])
        action_description_info = descriptions["predicates"][action_name]
        action_description, arg_order = action_description_info.split('#')
        action_str = f'({action_name} ' + ' '.join(ground_objs) + ')'
        action = self.action_preds[action_name]( *[ground_objs[int(index)] for index in arg_order.strip()])
        return action
 


def main(problem_idx, max_actions):
    """Runs the LLM prompting program on a training episode for `max_actions` number of actions or until the goal is reached."""

    env = pddlgym.make("PDDLEnvBakingrealistic-v0")
    env.fix_problem_index(problem_idx)
    obs, _ = env.reset()

    llm_agent = LLMAgent(env)

    goal_reached = False
    num_actions = 0
    while not goal_reached and num_actions < max_actions:

        action = llm_agent.get_action(obs, problem_idx)
        print("Executing action ", action)
        next_obs, rew, episode_done, _ = env.step(action)
        positive_effects = {e for e in next_obs.literals - obs.literals}
        negative_effects = {Anti(ne) for ne in obs.literals - next_obs.literals}
        effects = positive_effects | negative_effects
        print("Effects: ", effects)
        obs = next_obs

        num_actions += 1

        if rew == 1.0:
            goal_reached = True
            print("Reached goal!")

if __name__ == '__main__':
    main(0, 1)