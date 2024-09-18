"""Prototype some prompt templates that gets correct plans to all train goals.

Agenda:
DONE 1. Create the prompting program usign ChatGPT website so that OpenAI GPT-4o can get the correct ground plan for problem 1.

2. Refactor the JSON for predicate descriptions and create a problems.json for each train goal description. 

3. Use the API instead of ChatGPT website to get plans for all the other train goals.

3. Test the prompting program with other LLMs. Try to get a set of prompts that get the correct plan for each LLM. 

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
parser = argparse.ArgumentParser()
parser.add_argument('--problem_file', type=str)
args = parser.parse_args()
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
with open('predicate_and_goal-descriptions.json', 'r') as f:
    descriptions = json.load(f)
 
initial_state_predicate_fstrings = get_facts(descriptions, initial_state)

goal_state_predicate_fstrings = descriptions['goal']

action_predicate_fstrings = get_example_ground_actions(descriptions, ground_actions_string)

intro_prompt = \
f"""
You are a household robot in a kitchen. You are in front of the kitchen counter, where there are some prepared ingredients. Your task is to decide the sequence of actions to bake desserts.

More specifically, you will be given a set of facts that are currently true in the world, and a set of facts that is your goal to make true in the world. With my step-by-step guidance, you will think through how to act to achieve the goal.
"""
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

actions_prompt = \
f"""Now that you know where you are starting from and where you'd like to get to, I'll tell you about the possible things you can do in the kitchen. Then, your job will be to find a sequence of these things that takes you from the start to the goal.

Here are examples of the actions you know how to do. These are pre-defined atomic actions that you can execute.

{action_predicate_fstrings}
"""

start_thinking_prompt = \
f"""Now that you know what actions you can take to manipulate the environment, let's think through some plans that will get us to the goal from our initial state.
"""

#[Optional]: optionally give more details at the starting state than the predicate descriptions: e.g. the butter is one lump of how many tablespoons, etc.

intro = ""
for prompt in [intro_prompt, problem_setting_prompt, start_thinking_prompt]:
    intro += prompt
print("**********************PROMPT********************")
print(intro)
input()

import json
with open('problem1-descriptions.json', 'r') as f:
    descriptions = json.load(f)

action_description_string = ""
for k, v in descriptions["lifted_skill_descriptions"].items():
    action_description_string += k + ": " + v + '\n'
formalizing_intro = \
f"""Now that we have a high level sketch of how to get to our goal, let's formalize these steps into a formatted plan.

These are the names of the atomic actions that we can perform, along with their descriptions:
{action_description_string}

Can you please give a sequence of these phrases that will get us to the goal? Format it using a numbered list with one line per step, starting with "1.".
"""

print("**********************PROMPT********************")
print(formalizing_intro)
input("Paste the parsed plan steps / actions into the script")

#TODO: parse the plan
parsed_plan = \
"""
1. pour-powdery-ingredient-from-measuring-cup

Objects involved: flour-0, measuring-cup-0, bowl-0
Flour-0 is transferred from measuring-cup-0 to bowl-0.
2. pour-powdery-ingredient-from-measuring-cup

Objects involved: flour-1, measuring-cup-1, bowl-0
Flour-1 is transferred from measuring-cup-1 to bowl-0.
3. pour-powdery-ingredient-from-measuring-cup

Objects involved: baking-powder-0, measuring-cup-2, bowl-0
Baking-powder-0 is transferred from measuring-cup-2 to bowl-0.
4. pour-powdery-ingredient-from-measuring-cup

Objects involved: sugar-0, measuring-cup-3, bowl-0
Sugar-0 is transferred from measuring-cup-3 to bowl-0.
5. put-butter-in-container-from-measuring-cup

Objects involved: butter-0, measuring-cup-4, bowl-0
Butter-0 is transferred from measuring-cup-4 to bowl-0.
6. crack-egg-and-put-in-container

Objects involved: egg-0, bowl-0
Egg-0 is cracked and placed in bowl-0; the shell is discarded.
7. crack-egg-and-put-in-container

Objects involved: egg-1, bowl-0
Egg-1 is cracked and placed in bowl-0; the shell is discarded.
8. use-stand-mixer

Objects involved: bowl-0, mixture-0
The electric stand mixer is used on bowl-0 to create mixture-0, which contains all ingredients in the bowl.
9. pour-mixture-only

Objects involved: mixture-0, bowl-0, pan-0
Mixture-0 is poured from bowl-0 into pan-0.
10. preheat-oven-with-cake-settings

Objects involved: oven
The oven is preheated to 350 degrees Fahrenheit for cake baking.
11. put-container-in-oven

Objects involved: pan-0, oven
Pan-0 (containing mixture-0) is placed inside the oven.
12. start-baking-with-cake-settings

Objects involved: oven, pan-0, mixture-0
The oven is set to bake pan-0 (with mixture-0) for the required cake-baking time.
13. remove-pan-from-oven

Objects involved: pan-0, dessert-0 (once baked), oven
Pan-0 (now containing dessert-0, the cake) is removed from the oven and placed on the counter to cool.
"""

#TODO: replace these with the parsed actions
action_names = [
"pour-powdery-ingredient-from-measuring-cup",
"pour-powdery-ingredient-from-measuring-cup",
"pour-powdery-ingredient-from-measuring-cup",
"pour-powdery-ingredient-from-measuring-cup",
"put-butter-in-container-from-measuring-cup",
"crack-egg-and-put-in-container",
"crack-egg-and-put-in-container",
"use-stand-mixer",
"pour-mixture-only",
"preheat-oven-with-cake-settings",
"put-container-in-oven",
"start-baking-with-cake-settings",
"remove-pan-from-oven",
]
 
grounding_plan_prompt = \
"""Thanks. First let's think step by step what objects are associated with each of these actions, and then we'll try the plan to get to the goal state.
"""
print("**********************PROMPT********************")
print(grounding_plan_prompt)
input()

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
f"""Let's recap what we've talked about. Currently, the following facts are true:

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