"""Prototype some prompt templates that gets correct plans to all train goals.

Agenda:
DONE 1. Create the prompting program usign ChatGPT website so that OpenAI GPT-4o can get the correct ground plan for problem 1.

DONE 2. Refactor the JSON for predicate descriptions and create a problems.json for each train goal description. 

DONE 3. Use the ChatGPT website to get plans for all the other train goals.

4. Try to recover the program that solved train task 4.

5. Get a prompting program that solves train tasks 1-3 a little reliably. Try the first couple of prompts a few times in the website to get the right plan sketches.

6. Use the API to verify with more experiments that the prompting program at least solves tasks 1-3.

7. Try to find a prompting program that will solve train goal 4.

Things to try:
a) Give context of the oven capacity.
b) Give context about the object descriptions related to each action before asking for the step-by-step plan.
c) When ask about a step, and selecting objects, remind about actions performed already that involve each object. (May need to edit the arity of action predicates for this)
d) Use modified context for certain prompts, instead of the whole conversation history.

8. Code up using API to prototype with partial conversations. Have a flag to use interactive / ChatGPT website mode, or use API.


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

You should have all of the ingredients that you need on the counter prepared for you. Please make sure that each dessert you make has enough real, non-hypothetical ingredients allocated to it. If making multiple desserts, please make sure that you don't allocate the same ingredients between desserts. If you are strained to get enough ingredients for all the desserts you are making, it's okay to use less of certain ingredients: as long as each dessert has all the necessary ingredient types, it will be fine.
"""
print("**********************PROMPT********************")
print(problem_setting_prompt)
input()

actions_prompt = \
f"""Now that you know where you are starting from and where you'd like to get to, I'll tell you about the possible things you can do in the kitchen. Then, your job will be to find a sequence of these things that takes you from the start to the goal.

Here are examples of the actions you know how to do. These are pre-defined atomic actions that you can execute.

""" + '\n'.join(action_predicate_fstrings) + '\n\n'

start_thinking_prompt = \
f"""Now that you know what actions you can take to manipulate the environment, let's think through some plans that will get us to the goal from our initial state. If you are baking desserts, please rationalize what are the essential ingredients and their amounts to make those desserts and use only those. Please provide the full plan for each dessert.
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

Can you please give a sequence of these phrases that will get us to the goal? Format it using a numbered list with one line per step, starting with "1.". Give a little explanation of each step underneath each bullet point. Mark the end of the plan with '***' in your response.
"""

print("**********************PROMPT********************")
print(formalizing_intro)
input()

print("**********************OPTIONAL PROMPT********************")
mark_end_of_plan_prompt = \
"""Please mark the end of the plan with '***' in your response.
"""
print(mark_end_of_plan_prompt)
input()

print("**********************OPTIONAL PROMPT********************")
#TODO: parse the actions, and if any of them can't be looked up, prompt again with the following:
# TODO: when parsing the action string, look for the name in the whole bullet point, not just the text following "#."
print(f"Step i doesn't use any of the phrases below. Could you please revise the plan using only phrases from the following list? Write them exactly how they appear in the list. Give a little explanation of each step underneath each bullet point.\n{action_names_string}")
input("Paste the parsed plan steps / actions into the script")

#TODO: parse the plan
#TODO: make sure this string starts each bulletpoint with a newline then 1.: `\n1.`
#TODO: make sure that this contains up to the last bulletpoint information, and no more.
parsed_plan = """
Here is the formalized plan to achieve the goal:

### **Plan for Cake (dessert-0)**

1. **crack-egg-and-put-in-container (egg-0, bowl-0)**
   - Crack the first egg (egg-0) into bowl-0, discarding the shell.

2. **crack-egg-and-put-in-container (egg-2, bowl-0)**
   - Crack the second egg (egg-2) into bowl-0, discarding the shell.

3. **pour-powdery-ingredient-from-measuring-cup (sugar-1, measuring-cup-6, bowl-0)**
   - Pour all of the sugar-1 from measuring-cup-6 into bowl-0.

4. **pour-powdery-ingredient-from-measuring-cup (flour-1, measuring-cup-1, bowl-0)**
   - Pour all of the flour-1 from measuring-cup-1 into bowl-0.

5. **put-butter-in-container-from-measuring-cup (butter-0, measuring-cup-4, bowl-0)**
   - Add all of the butter-0 from measuring-cup-4 into bowl-0.

6. **pour-powdery-ingredient-from-measuring-cup (baking-powder-0, measuring-cup-2, bowl-0)**
   - Pour all of the baking-powder-0 from measuring-cup-2 into bowl-0.

7. **use-stand-mixer (bowl-0)**
   - Use the stand mixer to combine all ingredients in bowl-0 until smooth.

8. **preheat-oven-with-cake-settings**
   - Preheat the oven to 350°F, the temperature to bake a cake.

9. **pour-mixture-only (bowl-0, pan-0)**
   - Pour the cake batter mixture from bowl-0 into pan-0.

10. **put-pan-or-bowl-in-oven (pan-0)**
    - Place pan-0, which contains the cake batter, into the preheated oven.

11. **start-baking-with-cake-settings**
    - Set the time and start baking the cake with cake settings.

---

### **Plan for Soufflé (dessert-1)**

12. **crack-egg-and-put-in-container (egg-4, bowl-1)**
    - Crack the first egg (egg-4) into bowl-1, discarding the shell.

13. **crack-egg-and-put-in-container (egg-5, bowl-1)**
    - Crack the second egg (egg-5) into bowl-1, discarding the shell.

14. **separate-raw-yolk-from-egg-whites (egg-4, bowl-1)**
    - Separate egg-4 into yolk and egg whites, keeping them in different containers.

15. **separate-raw-yolk-from-egg-whites (egg-5, bowl-1)**
    - Separate egg-5 into yolk and egg whites, keeping them in different containers.

16. **beat-egg-whites (bowl-1)**
    - Beat the egg whites from bowl-1 using the electric mixer until stiff peaks form.

17. **pour-powdery-ingredient-from-measuring-cup (sugar-0, measuring-cup-3, bowl-0)**
    - Pour all of the sugar-0 from measuring-cup-3 into bowl-0.

18. **pour-powdery-ingredient-from-measuring-cup (flour-0, measuring-cup-0, bowl-0)**
    - Pour all of the flour-0 from measuring-cup-0 into bowl-0.

19. **put-butter-in-container-from-measuring-cup (butter-1, measuring-cup-5, bowl-0)**
    - Add all of the butter-1 from measuring-cup-5 into bowl-0.

20. **use-stand-mixer (bowl-0)**
    - Use the stand mixer to mix the sugar, flour, and butter into a smooth base.

21. **fold-stiff-egg-whites-into-mixture (bowl-1, bowl-0)**
    - Fold the stiff egg whites from bowl-1 into the soufflé mixture in bowl-0 using a spatula.

22. **preheat-oven-with-souffle-settings**
    - Preheat the oven to 375°F, the temperature to bake a soufflé.

23. **pour-mixture-only (bowl-0, pan-1)**
    - Pour the soufflé mixture from bowl-0 into pan-1.

24. **put-pan-or-bowl-in-oven (pan-1)**
    - Place pan-1, which contains the soufflé batter, into the preheated oven.

25. **start-baking-with-souffle-settings**
    - Set the time and start baking the soufflé with soufflé settings.

***

This plan covers the required actions for making both a cake and a soufflé, ensuring that all steps are followed correctly to reach the desired goal.
"""

#TODO: replace these with the parsed actions
action_names = [
    # cake plan
"crack-egg-and-put-in-container",
"crack-egg-and-put-in-container",
"pour-powdery-ingredient-from-measuring-cup",
"pour-powdery-ingredient-from-measuring-cup",
"put-butter-in-container-from-measuring-cup",
"pour-powdery-ingredient-from-measuring-cup",
"use-stand-mixer",
"preheat-oven-with-cake-settings",
"pour-mixture-only",
"put-pan-or-bowl-in-oven",
"start-baking-with-cake-settings",
# souffle plan
"crack-egg-and-put-in-container",
"crack-egg-and-put-in-container",
"separate-raw-yolk-from-egg-whites",
"separate-raw-yolk-from-egg-whites",
"beat-egg-whites",
"pour-powdery-ingredient-from-measuring-cup",
"pour-powdery-ingredient-from-measuring-cup",
"put-butter-in-container-from-measuring-cup",
"use-stand-mixer",
"fold-stiff-egg-whites-into-mixture",
"preheat-oven-with-souffle-settings",
"pour-mixture-only",
"put-pan-or-bowl-in-oven",
"start-baking-with-souffle-settings",
]
 
def get_objects_of_type(obj_string, object_type):
    objs = []
    for line in obj_string.split('\n'):
        if line.strip() == '': continue
        name, obj_type = line.split(' - ') 
        if obj_type.strip() == object_type:
            objs.append(name)
    return objs

import re

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
    if len(actions_list) + 2 <= len(action_names):
        match = re.search(f'\n{len(actions_list)+1}\.[\s\S.]*\n{len(actions_list)+2}\.', parsed_plan)
        step = parsed_plan[match.start() : match.end() - len(str(len(actions_list) + 2) + '.')]
    else:
        match = re.search(f'\n{len(actions_list)+1}\.[\s\S.]*', parsed_plan)
        step = parsed_plan[match.start(): match.end()]
    step = step.strip()
    reminder_prompt = \
f"""As a reminder, we are doing the following step in the plan sketch. You may need to revise the step if it's incorrect:

{step}

"""
    action_arg_prompt += \
f"""
Okay, now, we are thinking about step {len(actions_list) + 1} in our plan sketch. We are going to do the following action: {action_description_with_nonspecific_articles[:-1].lower()}. We need to identify the names of the specific objects involved in this action. Here are more details about how the objects involved need to relate to the action:
""" + '\n'.join(variable_description_list) + '\n' + reminder_prompt
    print(action_arg_prompt)

    ground_objs = []
    for i, variable_description in enumerate(variable_description_list):
        object_type = action_preds[action_name].var_types[i]
        objects_list = get_objects_of_type(OBJECTS_STRING, object_type)

        if len(objects_list) == 1:
            ground_objs.append(objects_list[0])
        else:
            action_grounding_variable_prompt = \
f"""We are going to {action_description_with_nonspecific_articles[:-1].lower()}. Given knowledge of the current state and our planned actions, which of the following objects fits the description, {variable_description}?
""" + '\n'.join(objects_list) + '\n' + 'Please explain your answer, and then answer with the object name on the last line after "Answer:".'
            print(action_grounding_variable_prompt)
            #TODO: parse the object name after `Answer:`
            obj_name = input("Object:")
            ground_objs.append(obj_name)
    action_description_info = descriptions["predicates"][action_name]
    action_description, arg_order = action_description_info.split('#')
    actions_list.append(action_description.strip().format(*[ground_objs[int(index)] for index in arg_order.strip()]))
    #[Optional]: print the current `parsed_plan` and ask it to update the steps given this action description on the current step.
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