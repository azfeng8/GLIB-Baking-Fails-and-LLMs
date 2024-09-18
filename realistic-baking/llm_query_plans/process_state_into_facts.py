state = """
(is-pan pan-0)
(is-pan pan-1)
(is-bowl bowl-0)
(is-bowl bowl-1)
(is-plate plate-0)
(dessert-is-hypothetical dessert-0)
(is-egg egg-0)
(is-in-shell egg-0)
(is-egg egg-1)
(is-in-shell egg-1)
(egg-is-hypothetical egg-2)
(egg-is-hypothetical egg-3)
(is-tablespoons-of-flour flour-0)
(powder-ingredient-in-measuring-cup flour-0 measuring-cup-0)
(is-cups-of-flour flour-0)
(powder-ingredient-in-measuring-cup flour-1 measuring-cup-1)
(is-baking-powder baking-powder-0)
(powder-ingredient-in-measuring-cup baking-powder-0 measuring-cup-2)
(is-sugar sugar-0)
(powder-ingredient-in-measuring-cup sugar-0 measuring-cup-3)
(is-butter butter-0)
(butter-in-measuring-cup butter-0 measuring-cup-4)
(mixture-is-hypothetical mixture-0)
(mixture-is-hypothetical mixture-1)
(mixture-is-hypothetical mixture-2)
(mixture-is-hypothetical mixture-3)
(mixture-is-hypothetical mixture-4)
(mixture-is-hypothetical mixture-5)
(mixture-is-hypothetical mixture-6)
(mixture-is-hypothetical mixture-7)
(mixture-is-hypothetical mixture-8)
"""
# Translate each predicate into a factual statement, and tag it with the objects that it references.
import json
with open('problem1-descriptions.json', 'r') as f:
    descriptions = json.load(f)
    
# // Label the permutation of the arguments after the pound sign #.,

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
print(get_facts(descriptions, state))
