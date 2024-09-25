import itertools
import numpy as np
"""TODO: This script needs major update.

DONE use-stand-mixer now takes in a 3rd argument, which is the name of the final mixture item. This should always be hypothetical before the action.

The container can have the individual ingredients or mixtures inside. Assume that no more than 3 mixtures will be in a container at once. The desired behavior is that everything in the container will be contained in one `mixture` after mixing.

"""
def create_mix_in_pan_ops():
    action_name = 'use-stand-mixer-in-pan-with-mixture-set1'
    preconds_s = \
""":precondition (and
    (use-stand-mixer ?mixer ?pan)
    (is-pan ?pan)
    (mixture-is-hypothetical ?mixture)
"""
    effects_s = \
""":effect (and
    (pan-is-damaged ?pan)
    (not (mixture-is-airy ?mixture))"""

def create_mix_in_bowl_ops():
    object_types = [
        # 'powder_ingredient_hypothetical',
        'powder_ingredient_hypothetical',
        'powder_ingredient_hypothetical',
        'powder_ingredient_hypothetical',
        'butter_hypothetical',
    ]
    objects = [
        # 'tablespoons-of-flour',
        'cups-of-flour',
        'sugar',
        'baking-powder',
        'butter',
    ]
    egg_combos = [
        # ('whole-raw-egg', 'whole-raw-egg',),
        # ('raw-egg-whites', 'raw-egg-whites'),
        # ('raw-egg-yolk', 'raw-egg-yolk'),
        # ('raw-egg-yolk', 'hypothetical-egg'),
        ('whole-raw-egg', 'hypothetical-egg'),
        # ('raw-egg-whites', 'hypothetical-egg'),
    ]
    params_s = ':parameters (?mixer - electric_stand_mixer ?bowl - container ?mixture - mixture_hypothetical ?egg1 - egg_hypothetical ?egg2 - egg_hypothetical ?cupsofflour - powder_ingredient_hypothetical ?tablespoonsofflour - powder_ingredient_hypothetical ?sugar - powder_ingredient_hypothetical ?bakingpowder - powder_ingredient_hypothetical ?butter - butter_hypothetical)'
    action_name = 'use-stand-mixer-in-bowl-with-mixture-set1'
    objects_and_types = list(zip(objects, object_types))
    preconds_s = \
""":precondition (and
    (use-stand-mixer ?mixer ?bowl ?mixture)
    (is-bowl ?bowl)
    (mixture-is-hypothetical ?mixture)"""
    effects_s = \
""":effect (and
    (not (mixture-is-airy ?mixture))
    (mixture-in-container ?bowl ?mixture)
    (is-mixture ?mixture)
    (not (mixture-is-hypothetical ?mixture))
    """
    n=0
    objects_and_types = list(zip(objects, object_types))

    n+= create_ops(objects_and_types, egg_combos, preconds_s, effects_s, params_s, action_name, 'bowl')

    print(f"Got {n} ops")


def create_ops(objects_and_types_without_eggs, egg_combos, preconds_s_init, effects_s_init, params_s_init, action_name_init, container='pan'):
    n = 0
    for eggs in egg_combos:
        objects_and_types = [(ob, o_type, ob.replace('-', '')) for ob, o_type in objects_and_types_without_eggs]
        objects_and_types.extend([(egg, 'egg_hypothetical', f'egg{i+1}') for i, egg in enumerate(eggs)])

        for i in range(1, len(objects_and_types) + 1):
            for o_set_indices in itertools.combinations(np.arange(len(objects_and_types)), i, ):
                o_set = [objects_and_types[i] for i in o_set_indices]
                not_in_bowl_o_set = [objects_and_types[i] for i in np.arange(len(objects_and_types)) if i not in o_set_indices]
                params_s = params_s_init
                effects_s = effects_s_init
                preconds_s = preconds_s_init
                action_name = action_name_init

                o_set = sorted(o_set, key=lambda x: x[0])
                not_in_bowl_o_set = sorted(not_in_bowl_o_set, key=lambda x: x[0])

                preconds_set = set()
                effects_set = set()
                for ob, o_type, var in o_set:
                    prefix = o_type[:-len('_hypothetical')]
                    prefix = prefix.replace('_', '-')
                    if ob == 'hypothetical-egg':
                        preconds_set.add(f'(not (egg-in-container ?{container} ?{var}))')
                    else:
                        action_name += f'-{ob}'
                        # Preconds
                        preconds_set.add(f'(is-{ob} ?{var})')
                        preconds_set.add(f'({prefix}-in-container ?{container} ?{var})')
                        # Effect
                        effects_set.add(f'(not (is-{ob} ?{var}))')
                        effects_set.add(f'(mixture-has-{ob} ?mixture)')
                        effects_set.add(f'(not ({prefix}-in-container ?{container} ?{var}))')

                for ob, o_type, var in not_in_bowl_o_set:
                    prefix = o_type[:-len('_hypothetical')]
                    prefix = prefix.replace('_', '-')
                    if ob == 'hypothetical-egg':
                        preconds_set.add(f'(not (egg-in-container ?{container} ?{var}))')
                    else:
                        preconds_set.add(f'(not ({prefix}-in-container ?{container} ?{var}))')

                for eff in sorted(effects_set):
                    effects_s += f'\n    {eff}'

                for pre in sorted(preconds_set):
                    preconds_s += f'\n    {pre}'

                effects_s += '\n)'
                preconds_s += '\n)'

                operator_s = f'(:action {action_name}'
                operator_s += '\n' + params_s
                operator_s += '\n' + preconds_s + '\n' + effects_s + '\n)'
                print(operator_s)
                n+=1
    return n
# create_mix_in_pan_ops()
create_mix_in_bowl_ops()