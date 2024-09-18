import itertools
"""TODO: This script needs major update.

use-stand-mixer now takes in a 3rd argument, which is the name of the final mixture item. This should always be hypothetical before the action.

The container can have the individual ingredients or mixtures inside. Assume that no more than 3 mixtures will be in a container at once. The desired behavior is that everything in the container will be contained in one `mixture` after mixing.

"""
def create_mix_in_pan_ops():
    object_types = ['egg_hypothetical', 'egg_hypothetical', 'powder_ingredient_hypothetical', 'butter_hypothetical', 'powder_ingredient_hypothetical', 'powder_ingredient_hypothetical',]
    objects = ['raw-egg-yolk',  'raw-egg-whites',  'tablespoons-of-flour', 'butter', 'sugar', 'baking-powder']
    objects_and_types = list(zip(objects, object_types))
    params_s = ':parameters (?mixer - electric_stand_mixer ?pan - container ?mixture - mixture_hypothetical '
    action_name = 'use-stand-mixer-in-pan-with-mixture'
    preconds_s = \
""":precondition (and
    (use-stand-mixer ?mixer ?pan)
    (mixture-in-container ?pan ?mixture)
    (is-pan ?pan)
    (not (mixture-is-hypothetical ?mixture))
    (is-mixture ?mixture)"""
    effects_s = \
""":effect (and
    (pan-is-damaged ?pan)
    (not (mixture-is-airy ?mixture))"""
    n=0
    n+= create_ops(objects_and_types, preconds_s, effects_s, params_s, action_name)

    # Get the ops with more flour instead of less
    objects = ['raw-egg-yolk',  'raw-egg-whites',  'cups-of-flour', 'butter', 'sugar', 'baking-powder']
    object_types = ['egg_hypothetical', 'egg_hypothetical', 'powder_ingredient_hypothetical', 'butter_hypothetical', 'powder_ingredient_hypothetical', 'powder_ingredient_hypothetical',]
    objects_and_types = list(zip(objects, object_types))
    n+= create_ops(objects_and_types, preconds_s, effects_s, params_s, action_name)

    # Get the ops for making a new mixture
    action_name = 'use-stand-mixer-in-pan-with'
    preconds_s = \
""":precondition (and
    (use-stand-mixer ?mixer ?pan)
    (is-pan ?pan)
    (mixture-is-hypothetical ?mixture)"""
    effects_s = \
""":effect (and
    (pan-is-damaged ?pan)
    (is-mixture ?mixture)
    (not (mixture-is-hypothetical ?mixture))
    (mixture-in-container ?pan ?mixture)
    (not (mixture-is-airy ?mixture))"""
    n+= create_ops(objects_and_types, preconds_s, effects_s, params_s, action_name)

    objects = ['raw-egg-yolk',  'raw-egg-whites',  'tablespoons-of-flour', 'butter', 'sugar', 'baking-powder']
    object_types = ['egg_hypothetical', 'egg_hypothetical', 'powder_ingredient_hypothetical', 'butter_hypothetical', 'powder_ingredient_hypothetical', 'powder_ingredient_hypothetical',]
    objects_and_types = list(zip(objects, object_types))
    n+= create_ops(objects_and_types, preconds_s, effects_s, params_s, action_name)

    # Get the ops with whole egg mixed in to get mixture with egg yolk and whites
    objects = ['cups-of-flour', 'butter', 'sugar', 'baking-powder']
    object_types = ['powder_ingredient_hypothetical', 'butter_hypothetical', 'powder_ingredient_hypothetical', 'powder_ingredient_hypothetical',]
    objects_and_types = list(zip(objects, object_types))
    action_name = 'use-stand-mixer-in-pan-with-mixture-wholeegg'
    preconds_s = \
""":precondition (and
    (use-stand-mixer ?mixer ?pan)
    (mixture-in-container ?pan ?mixture)
    (is-pan ?pan)
    (not (mixture-is-hypothetical ?mixture))
    (is-mixture ?mixture)
    (is-whole-raw-egg ?wholerawegg)
    (egg-in-container ?pan ?wholerawegg)"""
    effects_s = \
""":effect (and
    (pan-is-damaged ?pan)
    (not (egg-in-container ?pan ?wholerawegg))
    (not (is-whole-raw-egg ?wholerawegg))
    (mixture-has-raw-egg-yolk ?mixture)
    (mixture-has-raw-egg-whites ?mixture)
    (not (mixture-is-airy ?mixture))"""
    params_s = ':parameters (?mixer - electric_stand_mixer ?pan - container ?mixture - mixture_hypothetical ?wholerawegg - egg_hypothetical '
    n += create_ops(objects_and_types, preconds_s, effects_s, params_s, action_name)

    objects = ['tablespoons-of-flour', 'butter', 'sugar', 'baking-powder']
    object_types = ['powder_ingredient_hypothetical', 'butter_hypothetical', 'powder_ingredient_hypothetical', 'powder_ingredient_hypothetical',]
    objects_and_types = list(zip(objects, object_types))
    n+=  create_ops(objects_and_types, preconds_s, effects_s, params_s, action_name)

    # With new mixture
    action_name = 'use-stand-mixer-in-pan-with-wholeegg'
    preconds_s = \
""":precondition (and
    (use-stand-mixer ?mixer ?pan)
    (is-pan ?pan)
    (mixture-is-hypothetical ?mixture)
    (is-whole-raw-egg ?wholerawegg)
    (egg-in-container ?pan ?wholerawegg)"""
    effects_s = \
""":effect (and
    (pan-is-damaged ?pan)
    (is-mixture ?mixture)
    (not (mixture-is-hypothetical ?mixture))
    (not (is-whole-raw-egg ?wholerawegg))
    (mixture-has-raw-egg-yolk ?mixture)
    (mixture-has-raw-egg-whites ?mixture)
    (mixture-in-container ?pan ?mixture)
    (not (egg-in-container ?pan ?wholerawegg))
    (not (mixture-is-airy ?mixture))"""
    n+=  create_ops(objects_and_types, preconds_s, effects_s, params_s, action_name)

    objects = ['cups-of-flour', 'butter', 'sugar', 'baking-powder']
    object_types = ['powder_ingredient_hypothetical', 'butter_hypothetical', 'powder_ingredient_hypothetical', 'powder_ingredient_hypothetical',]
    objects_and_types = list(zip(objects, object_types))
    n+=  create_ops(objects_and_types, preconds_s, effects_s, params_s, action_name)
    # print(f"Got {n} ops")

def create_mix_in_bowl_ops():
    object_types = ['egg_hypothetical', 'egg_hypothetical', 'powder_ingredient_hypothetical', 'butter_hypothetical', 'powder_ingredient_hypothetical', 'powder_ingredient_hypothetical',]
    objects = ['raw-egg-yolk',  'raw-egg-whites',  'tablespoons-of-flour', 'butter', 'sugar', 'baking-powder']
    objects_and_types = list(zip(objects, object_types))
    params_s = ':parameters (?mixer - electric_stand_mixer ?bowl - container ?mixture - mixture_hypothetical '
    action_name = 'use-stand-mixer-in-bowl-with-mixture'
    preconds_s = \
""":precondition (and
    (use-stand-mixer ?mixer ?bowl)
    (mixture-in-container ?bowl ?mixture)
    (is-bowl ?bowl)
    (not (mixture-is-hypothetical ?mixture))
    (is-mixture ?mixture)"""
    effects_s = \
""":effect (and
    (not (mixture-is-airy ?mixture))"""
    n=0
    n+= create_ops(objects_and_types, preconds_s, effects_s, params_s, action_name, 'bowl')

    # Get the ops with more flour instead of less
    objects = ['raw-egg-yolk',  'raw-egg-whites',  'cups-of-flour', 'butter', 'sugar', 'baking-powder']
    object_types = ['egg_hypothetical', 'egg_hypothetical', 'powder_ingredient_hypothetical', 'butter_hypothetical', 'powder_ingredient_hypothetical', 'powder_ingredient_hypothetical',]
    objects_and_types = list(zip(objects, object_types))
    n+= create_ops(objects_and_types, preconds_s, effects_s, params_s, action_name, 'bowl')

    # Get the ops for making a new mixture
    action_name = 'use-stand-mixer-in-bowl-with'
    preconds_s = \
""":precondition (and
    (use-stand-mixer ?mixer ?bowl)
    (is-bowl ?bowl)
    (mixture-is-hypothetical ?mixture)"""
    effects_s = \
""":effect (and
    (is-mixture ?mixture)
    (not (mixture-is-hypothetical ?mixture))
    (mixture-in-container ?bowl ?mixture)
    (not (mixture-is-airy ?mixture))"""
    n+= create_ops(objects_and_types, preconds_s, effects_s, params_s, action_name, 'bowl')

    objects = ['raw-egg-yolk',  'raw-egg-whites',  'tablespoons-of-flour', 'butter', 'sugar', 'baking-powder']
    objects_and_types = list(zip(objects, object_types))
    n+= create_ops(objects_and_types, preconds_s, effects_s, params_s, action_name, 'bowl')

    # Get the ops with whole egg mixed in to get mixture with egg yolk and whites
    objects = ['cups-of-flour', 'butter', 'sugar', 'baking-powder']
    object_types = ['powder_ingredient_hypothetical', 'butter_hypothetical', 'powder_ingredient_hypothetical', 'powder_ingredient_hypothetical',]
    objects_and_types = list(zip(objects, object_types))
    action_name = 'use-stand-mixer-in-bowl-with-mixture-wholeegg'
    preconds_s = \
""":precondition (and
    (use-stand-mixer ?mixer ?bowl)
    (mixture-in-container ?bowl ?mixture)
    (is-bowl ?bowl)
    (not (mixture-is-hypothetical ?mixture))
    (is-mixture ?mixture)
    (is-whole-raw-egg ?wholerawegg)
    (egg-in-container ?bowl ?wholerawegg)"""
    effects_s = \
""":effect (and
    (not (egg-in-container ?bowl ?wholerawegg))
    (not (is-whole-raw-egg ?wholerawegg))
    (mixture-has-raw-egg-yolk ?mixture)
    (mixture-has-raw-egg-whites ?mixture)
    (not (mixture-is-airy ?mixture))"""
    params_s = ':parameters (?mixer - electric_stand_mixer ?bowl - container ?mixture - mixture_hypothetical ?wholerawegg - egg_hypothetical '
    n += create_ops(objects_and_types, preconds_s, effects_s, params_s, action_name, 'bowl')

    objects = ['tablespoons-of-flour', 'butter', 'sugar', 'baking-powder']
    object_types = ['powder_ingredient_hypothetical', 'butter_hypothetical', 'powder_ingredient_hypothetical', 'powder_ingredient_hypothetical',]
    objects_and_types = list(zip(objects, object_types))
    n+=  create_ops(objects_and_types, preconds_s, effects_s, params_s, action_name, 'bowl')

    # With new mixture
    action_name = 'use-stand-mixer-in-bowl-with-wholeegg'
    preconds_s = \
""":precondition (and
    (use-stand-mixer ?mixer ?bowl)
    (is-bowl ?bowl)
    (mixture-is-hypothetical ?mixture)
    (is-whole-raw-egg ?wholerawegg)
    (egg-in-container ?bowl ?wholerawegg)"""
    effects_s = \
""":effect (and
    (is-mixture ?mixture)
    (not (egg-in-container ?bowl ?wholerawegg))
    (not (mixture-is-hypothetical ?mixture))
    (not (is-whole-raw-egg ?wholerawegg))
    (mixture-has-raw-egg-yolk ?mixture)
    (mixture-has-raw-egg-whites ?mixture)
    (mixture-in-container ?bowl ?mixture)
    (not (mixture-is-airy ?mixture))"""
    n+=  create_ops(objects_and_types, preconds_s, effects_s, params_s, action_name, 'bowl')

    objects = ['cups-of-flour', 'butter', 'sugar', 'baking-powder']
    object_types = ['powder_ingredient_hypothetical', 'butter_hypothetical', 'powder_ingredient_hypothetical', 'powder_ingredient_hypothetical',]
    objects_and_types = list(zip(objects, object_types))
    n+=  create_ops(objects_and_types, preconds_s, effects_s, params_s, action_name, 'bowl')
    # print(f"Got {n} ops")


def create_ops(objects_and_types, preconds_s_init, effects_s_init, params_s_init, action_name_init, container='pan'):
    n = 0
    for i in range(1, len(objects_and_types) + 1):
        for o_set in itertools.combinations(objects_and_types, i, ):
            params_s = params_s_init
            effects_s = effects_s_init
            preconds_s = preconds_s_init
            action_name = action_name_init
            o_set = sorted(o_set, key=lambda x: x[0])
            for ob, o_type in o_set:
                var = ob.replace('-', '')
                action_name += f'-{ob}'
                params_s += f'?{var} - {o_type}' + ' '
                preconds_s += f'\n    (is-{ob} ?{var})'
            params_s = params_s[:-1]
            operator_s = f'(:action {action_name}'
            operator_s += '\n' + params_s + ')'
            for ob, o_type in o_set:
                var = ob.replace('-', '')
                prefix = o_type[:-len('_hypothetical')]
                prefix = prefix.replace('_', '-')
                preconds_s += f'\n    ({prefix}-in-container ?{container} ?{var})'
                # Effect
                effects_s += f'\n    (not (is-{ob} ?{var}))'
            for ob, o_type in o_set:
                # Effect
                effects_s += f'\n    (mixture-has-{ob} ?mixture)'
            for ob, o_type in o_set:
                var = ob.replace('-', '')
                prefix = o_type[:-len('_hypothetical')]
                prefix = prefix.replace('_', '-')
                effects_s += f'\n    (not ({prefix}-in-container ?{container} ?{var}))'
            effects_s += '\n)'
            preconds_s += '\n)'
            operator_s += '\n' + preconds_s + '\n' + effects_s + '\n)'
            print(operator_s)
            print("\n")
            n+=1
    return n
create_mix_in_pan_ops()
create_mix_in_bowl_ops()