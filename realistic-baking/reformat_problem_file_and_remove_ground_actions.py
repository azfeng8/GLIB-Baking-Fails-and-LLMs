skills_strings = \
"""
(beat-egg-whites ?arg0 - electric_stand_mixer ?arg1 - container ?arg2 - egg_hypothetical)
(crack-egg-and-put-in-container ?arg0 - egg_hypothetical ?arg1 - container)
(fold-stiff-egg-whites-into-mixture ?arg0 - spatula ?arg1 - container ?arg2 - container ?arg3 - egg_hypothetical)
(move-baked-good-in-container-to-different-container ?arg0 - container ?arg1 - container ?arg2 - dessert_hypothetical)
(pour-mixture-only ?arg0 - container ?arg1 - container ?arg2 - mixture_hypothetical)
(pour-powdery-ingredient-from-container ?arg0 - container ?arg1 - container ?arg2 - powder_ingredient_hypothetical)
(pour-powdery-ingredient-from-measuring-cup ?arg0 - powder_ingredient_hypothetical ?arg1 - measuring_cup ?arg2 - container)
(preheat-oven-with-cake-settings ?arg0 - oven)
(preheat-oven-with-souffle-settings ?arg0 - oven)
(put-butter-in-container-from-measuring-cup ?arg0 - butter_hypothetical ?arg1 - container)
(put-pan-in-oven ?arg0 - container ?arg1 - oven)
(remove-pan-from-oven ?arg0 - container)
(separate-raw-yolk-from-egg-whites ?arg0 - egg_hypothetical ?arg1 - egg_hypothetical ?arg2 - container ?arg3 - container)
(set-oven-with-cake-bake-time-and-press-start ?arg0 - oven ?arg1 - dessert_hypothetical)
(set-oven-with-souffle-bake-time-and-press-start ?arg0 - oven ?arg1 - dessert_hypothetical)
(transfer-butter-from-pan-or-bowl ?arg0 - container ?arg1 - container ?arg2 - butter_hypothetical)
(transfer-egg-from-pan-or-bowl ?arg0 - container ?arg1 - container ?arg2 - egg_hypothetical)
(use-stand-mixer ?arg0 - electric_stand_mixer ?arg1 - container ?arg2 - mixture_hypothetical)
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
args = parser.parse_args()
with open(args.file, 'r') as f:
    lines = f.readlines()
i = 0
line_to_insert = None
for line in lines:
    if ':init' in line:
        line_to_insert = i
    if line_to_insert is not None and line.strip() == ')':
        end_init = i 
        break
    i+=1

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
action_names = [a.name for a in actions]

init_state = []
ground_actions = []
for line in lines[line_to_insert:end_init]:
    items = line.strip()[1:-1] .split()
    name = items[0]
    if name in action_names:
        ground_actions.append(line)
    else:
        init_state.append(line)
init = init_state + [';all ground actions\n']

with open(args.file, 'w') as f:
    l = lines[:line_to_insert] + init + lines[end_init:]
    f.write(''.join(l))