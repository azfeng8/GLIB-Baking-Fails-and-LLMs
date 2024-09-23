"""Prototype some prompt templates that gets correct plans to all train goals.

Agenda:
DONE 1. Create the prompting program usign ChatGPT website so that OpenAI GPT-4o can get the correct ground plan for problem 1.

DONE 2. Refactor the JSON for predicate descriptions and create a problems.json for each train goal description. 

3. Use the ChatGPT website to get plans for all the other train goals.

4. Use the API to get the plans.

5. Test the prompting program with other LLMs. Try to get a set of prompts that get the correct plan for each LLM. 

"""
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

from pprint import pprint
def get_example_ground_actions(descriptions, ground_actions_string):
    max_num_examples_per_action = 2

    examples = []
    current_action_name = None
    ground_action_examples = []
    for line in ground_actions_string.split('\n'):
        line = line.strip()
        if line == '':
            continue
        items = line[1:-1].split()
        pred_name = items[0]
        if current_action_name is None or pred_name == current_action_name:
            ground_action_examples.append(line)
        else:
            examples.extend([ground_action_examples[i] for i in np.random.choice(len(ground_action_examples), min(max_num_examples_per_action, len(ground_action_examples)), replace=False)])
            ground_action_examples = [line]
        current_action_name = pred_name
    examples.extend([ground_action_examples[i] for i in np.random.choice(len(ground_action_examples), min(max_num_examples_per_action, len(ground_action_examples)), replace=False)])

    example_actions = ""
    for line in examples:
        items = line[1:-1].split()
        pred_name = items[0]
        description_string = descriptions["predicates"][pred_name]
        description, arg_order = description_string.split('#')
        description = description.strip()

        argument_order = items[1:]

        fstring_object_order = []
        for arg in arg_order.strip():
            fstring_object_order.append(argument_order[int(arg)])
        example_actions += (description.format(*fstring_object_order)) + '\n'
    return example_actions.strip()

# Parse the objects and initial state from the problem pddl file.
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--problem_file', type=str)
args = parser.parse_args()
problem_name = os.path.basename(args.problem_file)[:-len('.pddl')]
with open(args.problem_file, 'r') as f:
    lines = f.readlines()
line_start_idx = None
i = 0
for line in lines:
    if ':objects' in line:
        line_start_idx = i
    if line_start_idx is not None and ')' in line:
        OBJECTS_STRING = '\n'.join(l.strip() for l in lines[line_start_idx+1:i])
        break
    i += 1

line_start_idx = None
i = 0
for line in lines:
    if ":init" in line:
        line_start_idx = i
    if 'ground actions' in line:
        initial_state = '\n'.join(l.strip() for l in lines[line_start_idx+1:i])
    i+=1

line_start_idx = None
i = 0
for line in lines:
    if "ground actions" in line:
        line_start_idx = i
    if line_start_idx is not None and line.strip() == ')':
        ground_actions_string = '\n'.join(l.strip() for l in lines[line_start_idx+1:i])
        break
    i+=1
import numpy as np
import json
with open('predicate_and_goal_descriptions.json', 'r') as f:
    descriptions = json.load(f)
 
initial_state_predicate_fstrings = get_facts(descriptions, initial_state)

goal_state_predicate_fstrings = descriptions['train_goals'][problem_name]

action_predicate_fstrings = [f'{k}: {v}' for k, v in descriptions["lifted_skill_descriptions"].items()] #get_example_ground_actions(descriptions, ground_actions_string)

intro_prompt = \
f"""
You are a household robot in a kitchen. You are in front of the kitchen counter, where there are some prepared ingredients. Your task is to decide the sequence of actions to bake desserts.

More specifically, you will be given a set of facts that are currently true in the world, and a set of facts that is your goal to make true in the world. With my step-by-step guidance, you will think through how to act to achieve the goal.

Since you are baking desserts, first determine what are the differences between a cake and sweet, light, and airy souffle. Please rationalize what are the essential ingredients and their amounts to make those desserts and use only those.
"""
print("**********************PROMPT********************")
print(intro_prompt)
input()
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

These are the things that you would like to become true:
{goal_state_predicate_fstrings}

Getting to this state is your goal. We will spend the rest of the conversation trying to find the correct sequence of actions to get here.

"""
print("**********************PROMPT********************")
print(problem_setting_prompt)
input()

actions_prompt = \
f"""Now that you know where you are starting from and where you'd like to get to, I'll tell you about the possible things you can do in the kitchen. Then, your job will be to find a sequence of these things that takes you from the start to the goal.

Here are examples of the actions you know how to do. These are pre-defined atomic actions that you can execute.

""" + '\n'.join(action_predicate_fstrings) + '\n\n'

start_thinking_prompt = \
f"""Now that you know what actions you can take to manipulate the environment, let's think through some plans that will get us to the goal from our initial state. If you are baking desserts, please rationalize what are the essential ingredients and their amounts to make those desserts and use only those. Please provide the full plan.
"""

#[Optional]: optionally give more details at the starting state than the predicate descriptions: e.g. the butter is one lump of how many tablespoons, etc.

intro = ""
for prompt in [actions_prompt, start_thinking_prompt]:
    intro += prompt
print("**********************PROMPT********************")
print(intro)
input()

print("**********************PROMPT********************")
print("""At each step that you use the mixer, please list the ingredients that you will use in the mixing. If you don't need to use the mixer, then ignore this prompt. Make sure to not introduce any new actions that weren't in the list provided.""")
input()

action_description_string = ""
action_names_string = ""
for k, v in descriptions["lifted_skill_descriptions"].items():
    action_description_string += k + ": " + v + '\n'
    action_names_string += k  + '\n'
formalizing_intro = \
f"""Now that we have a high level sketch of how to get to our goal, let's formalize these steps into a formatted plan.

These are the names of the atomic actions that we can perform, along with their descriptions:
{action_description_string}

Can you please give a sequence of these phrases that will get us to the goal? Format it using a numbered list with one line per step, starting with "1.". Give a little explanation of each step underneath each bullet point.
"""

print("**********************PROMPT********************")
print(formalizing_intro)
input()
print("**********************OPTIONAL PROMPT********************")
#TODO: parse the actions, and if any of them can't be looked up, prompt again with the following:
# TODO: when parsing the action string, look for the name in the whole bullet point, not just the text following "#."
print(f"Step i doesn't use any of the phrases below. Could you please revise the plan using only phrases from the following list? Write them exactly how they appear in the list. Give a little explanation of each step underneath each bullet point.\n{action_names_string}")
input("Paste the parsed plan steps / actions into the script")

#TODO: parse the plan
parsed_plan = \
"""
1. pour-powdery-ingredient-from-measuring-cup

From: measuring-cup-0 (flour-0)
To: bowl-0
Pour the flour into bowl-0 to prepare the dry ingredients.
2. pour-powdery-ingredient-from-measuring-cup

From: measuring-cup-2 (baking-powder-0)
To: bowl-0
Add baking powder to bowl-0 for leavening the cake.
3. put-butter-in-container-from-measuring-cup

From: measuring-cup-4 (butter-0)
To: bowl-1
Transfer the butter to bowl-1 for creaming with sugar.
4. pour-powdery-ingredient-from-measuring-cup

From: measuring-cup-3 (sugar-0)
To: bowl-1
Add sugar to bowl-1 to combine with butter for a sweet base.
5. use-stand-mixer

In: bowl-1
Mix the butter and sugar together until light and fluffy.
6. crack-egg-and-put-in-container

To: bowl-1
Add egg-0 to the creamed mixture for moisture and richness.
7. pour-mixture-only

From: bowl-1
To: bowl-0
Combine the creamed mixture with the dry ingredients in bowl-0.
8. use-stand-mixer

In: bowl-0
Mix all ingredients in bowl-0 until you have a smooth batter.
9. pour-mixture-only

From: bowl-0
To: pan-0
Transfer the cake batter into pan-0, ready for baking.
10. preheat-oven-with-cake-settings

Preheat the oven to 350°F, the ideal temperature for baking cakes.
11. put-container-in-oven

Container: pan-0
Place the pan-0 in the preheated oven to bake the cake.
12. start-baking-with-cake-settings

Set the baking time for the cake and start the oven timer.
13. remove-pan-from-oven

Once baking is complete, carefully take the cake out of the oven to cool.
14. move-baked-good-in-container-to-different-container

From: pan-0
To: plate-0
Transfer the baked cake from pan-0 to plate-0 for serving.
"""

#TODO: replace these with the parsed actions
action_names = [
"pour-powdery-ingredient-from-measuring-cup",
"pour-powdery-ingredient-from-measuring-cup",
"put-butter-in-container-from-measuring-cup",
"pour-powdery-ingredient-from-measuring-cup",
"use-stand-mixer",
"crack-egg-and-put-in-container",
"pour-mixture-only",
"use-stand-mixer",
"pour-mixture-only",
"preheat-oven-with-cake-settings",
"put-container-in-oven",
"start-baking-with-cake-settings",
"remove-pan-from-oven",
"move-baked-good-in-container-to-different-container",
]
 
def get_objects_of_type(obj_string, object_type):
    objs = []
    for line in obj_string.split('\n'):
        if line.strip() == '': continue
        name, obj_type = line.split(' - ') 
        if obj_type.strip() == object_type:
            objs.append(name)
    return objs

import gym
import pddlgym
env = pddlgym.make("PDDLEnvBakingrealistic")
action_preds = {p.name: p for p in env.action_space.predicates}
 
actions_list = [
]
for action_name in action_names[len(actions_list):]:
    action_description_with_nonspecific_articles = descriptions["lifted_skill_descriptions"][action_name]
    variable_description_list = descriptions["skill_variable_descriptions"][action_name]
    print("**********************PROMPT********************")
    action_arg_prompt = \
f"""Thanks. Let's think step by step what objects are associated with each of these actions.
Let's recap what we've talked about. Currently, the following facts are true:

{initial_state_predicate_fstrings}

We want to make these facts true:
{goal_state_predicate_fstrings}

We're thinking through a plan step-by-step to our goal. The plan sketch is:

{parsed_plan}

"""

    if len(actions_list) > 0:
        actions_done_string = ""
        for i, action in enumerate(actions_list):
            actions_done_string += f'{i+1}. {action}\n'
            
        action_arg_prompt += \
f"""So far in our plan, we've already done:\n""" + actions_done_string

    action_arg_prompt += \
f"""
Okay, now, we are thinking about step {len(actions_list) + 1} in our plan sketch. We are going to do the following action: {action_description_with_nonspecific_articles[:-1].lower()}. We need to identify the names of the specific objects involved in this action. Here are more details about how the objects involved need to relate to the action:
""" + '\n'.join(variable_description_list) + '\n'
    print(action_arg_prompt)

    ground_objs = []
    for i, variable_description in enumerate(variable_description_list):
        object_type = action_preds[action_name].var_types[i]
        objects_list = get_objects_of_type(OBJECTS_STRING, object_type)

        if len(objects_list) == 1:
            ground_objs.append(objects_list[0])
        else:
            action_grounding_variable_prompt = \
f"""
We are going to {action_description_with_nonspecific_articles[:-1].lower()}. Given knowledge of the current state and our planned actions, which of the following objects fits the description, {variable_description}?
""" + '\n'.join(objects_list) + '\n' + 'Please explain your answer, and then answer with the object name on the last line after "Answer:".'
            print(action_grounding_variable_prompt)
            obj_name = input("Object:")
            ground_objs.append(obj_name)
    action_description_info = descriptions["predicates"][action_name]
    action_description, arg_order = action_description_info.split('#')
    actions_list.append(action_description.strip().format(*[ground_objs[int(index)] for index in arg_order.strip()]))
print('\n'.join(actions_list))

### [Optional] To review plans, ask for preconditions and effects, and important subgoals / milestones / a curriculum that would be good to achieve.
### [Optional] Only execute the first step of the plan, and then interactively execute the plan after each observation.

### [Optional] Do some chain-of-thought to get formalized plans.
# What are some subgoals that would help in reaching the goal?
# Give some example subgoals:
    # Give options using predicates and f-strings.
# ...extract the subgoals.

# How do we get to the [first subgoal]?
# ...append actions to the plan.
# Okay, say we started from [initial state], and executed [plan], and now we are at the [first subgoal]. How do we get from here to the [second subgoal]?
# ...append actions to the plan.

### Ask it to reflect on the plan, (maybe work backwards from the goal?)
# Okay, you helped give me a curriculum to reach the goal [end goal] from the initial state [start state]. You also gave me a plan to follow this curriculum. Now let's think step by step to refine the plan.

# What must be true to execute this action?
# What will happen after executing this action?
# According to the suggested plan [plan], [end state] will be the set of true facts after executing the action. Is this consistent with what we think will happen when executing the action [action]?
# If not, revise the plan. Otherwise, iterate through the plan to reflect on each action and predicted effects.