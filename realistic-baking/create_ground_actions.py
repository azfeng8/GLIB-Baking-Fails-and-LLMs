"""This script generates the list of ground actions needed in the problem PDDL file.

To use, replace the objects string with an object - object_type per line. Replace the skill strings as the declaration of the action predicates in the PDDL domain file.
"""

skills_strings = \
"""
(pour-powdery-ingredient-from-measuring-cup ?p - powder_ingredient_hypothetical ?cup - measuring_cup ?c - container)
(pour-mixture-only ?from - container ?into - container ?m - mixture_hypothetical)
(pour-powdery-ingredient-from-container ?from - container ?into - container ?m - powder_ingredient_hypothetical)
(transfer-butter-only ?from - container ?into - container ?m - butter_hypothetical)
(transfer-egg-only ?from - container ?into - container ?m - egg_hypothetical)
(move-baked-good-in-container-to-different-container ?from - container ?to - container)
(crack-egg-and-put-in-container ?e - egg_hypothetical ?c - container)
(put-butter-in-container-from-measuring-cup ?b - butter_hypothetical ?c - container)
(put-container-in-oven ?c - container ?o - oven)
(preheat-oven-with-cake-settings ?o - oven)
(preheat-oven-with-souffle-settings ?o - oven)
(use-stand-mixer ?m - electric_stand_mixer ?c - container ?m - mixture_hypothetical)
(remove-pan-from-oven ?c - container)
(start-baking-with-cake-settings ?o - oven)
(start-baking-with-souffle-settings ?o - oven)
(separate-raw-yolk-from-egg-whites ?e - egg_hypothetical ?new - egg_hypothetical)
(beat-egg-whites ?m - electric_stand_mixer ?c - container ?e - egg_hypothetical)
(fold-stiff-egg-whites-into-mixture ?s - spatula ?from - container ?to - container ?e - egg_hypothetical)
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
args = parser.parse_args()
with open(args.file, 'r') as f:
    lines = f.readlines()
line_start_idx = None
i = 0
for line in lines:
    if ':objects' in line:
        line_start_idx = i
    if line_start_idx is not None and ')' in line:
        objects_string = '\n'.join(l.strip() for l in lines[line_start_idx+1:i])
        break
    i += 1

from pprint import pprint
# Parse the objects list:
    # Create a map from object type to list of objects of that type.
objects = {}
for line in objects_string.split('\n'):
    if line == '': continue
    obj_name, obj_type = line.split(' - ')
    obj_type = obj_type.strip()
    obj_name = obj_name.strip()
    objects.setdefault(obj_type, [])
    objects[obj_type].append(obj_name)

# Parse the skills list:
    # Create a template Action with name, list of object types for its arguments
class Lifted_Action:
    def __init__(self, name, arg_object_types):
        self.name = name
        self.arg_object_types = arg_object_types
    def __repr__(self):
        return f'{self.name}(' + ','.join(self.arg_object_types) + ')'
    def __str__(self):
        return self.__repr__()

class Ground_Action:
    def __init__(self, name, objects):
        self.name = name
        self.objects = objects
    def __repr__(self) -> str:
        return f'({self.name} ' + ' '.join(self.objects) + ')'
    def __str__(self) -> str:
        return self.__repr__()

actions = []
for line in skills_strings.split('\n'):
    if line == '': continue
    line = line[1:-1]
    line_items =  line.split('?')
    action_name = line_items[0].strip()
    types = []
    for arg in line_items[1:]:
        if arg.strip() == '': continue
        _, arg_type = arg.split(' - ')
        arg_type = arg_type.strip()
        types.append(arg_type)
    actions.append(Lifted_Action(action_name.strip(), types))
# For each skill:
    # Count how many arguments of each type there are. Then do N_total_objects_of_type P N_args_of_that_type. Then use itertools.product between the permutations to get each enumerated arguments list.
    # in the innermost loop, iterate over the Action template's arguments, taking the next elt in the itertools.product of that argument type, to construct the final ground action predicate.
import itertools
n = 0
lines_to_insert = []
for action in actions:
    type_counts = {}
    for t in action.arg_object_types:
        type_counts.setdefault(t, 0)
        type_counts[t] += 1
    obj_type_permutations = []
    obj_types = []
    for obj_type, arg_count in type_counts.items():
        obj_type_permutations.append(list(itertools.permutations(objects[obj_type], arg_count)))
        obj_types.append(obj_type)
    for tuple_of_tuples in itertools.product(*obj_type_permutations):
        object_args = []
        objects_order_map = {}
        for obj_type, ground_objects in zip(obj_types, tuple_of_tuples):
            objects_order_map[obj_type] = list(ground_objects)
        for obj_type in action.arg_object_types:
            object_args.append(objects_order_map[obj_type].pop(0))
        ground_action = Ground_Action(action.name, object_args)
        lines_to_insert.append(f"        {ground_action}\n")
        n += 1
# print("Got ", n, "ground actions")
with open(args.file, 'r') as f:
    lines = f.readlines()
i = 0
line_to_insert = None
for line in lines:
    if 'ground actions' in line:
        line_to_insert = i
        break
    i+=1

i = 0
for line in lines:
    if ')' in line:
        prev_close_paran_i = i
    if 'goal' in line:
        line_end = prev_close_paran_i
        break
    i+=1
        
out_lines = lines[:line_to_insert+1] + lines_to_insert + lines[prev_close_paran_i:]
with open(args.file, 'w') as f:
    f.write(''.join(out_lines))

domain_actions_str = '; (:actions '
for action in actions:
    domain_actions_str += action.name + ' '
domain_actions_str = domain_actions_str[:-1]
domain_actions_str += ')'
# print(domain_actions_str)
        