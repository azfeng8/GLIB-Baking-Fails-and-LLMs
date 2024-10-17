import gym
import pddlgym
from pprint import pprint
from pddlgym.structs import Anti


# Load the plan
test = True
t = 'test' if test else 'train'
r = range(0, 20) if test else range(0,4)
problems = [(idx, f'{t}/problem{idx + 1}.txt') for idx in r]

for idx, fname in problems:
    print(f"*************************PROBLEM {idx + 1}*************************")
    with open(fname, 'r') as f:
        plan = f.readlines()

    if test:
        env = pddlgym.make('PDDLEnvBakingrealisticTest-v0')
    else:
        env = pddlgym.make('PDDLEnvBakingrealistic-v0')
    env.fix_problem_index(idx)
    obs, _ = env.reset()

    print(f"Make sure this is indeed problem {idx + 1} goal: ", obs.goal)
    input()

    actions = []
    for line in plan:
        if line.strip() == '': continue

        # print(line)
        items = line.strip()[1:-1].split()
        action_predicate_name = items[0]
        object_names = items[1:]
        action_pred = [p for p in env.action_space.predicates if p.name == action_predicate_name][0]
        objects = [o for o in obs.objects]
        args = []
        for object_name in object_names:
            for o in objects:
                obj_name, _ = o._str.split(':')
                if obj_name == object_name:
                    args.append(o)
                    break
        # print(action_pred, args)
        actions.append(action_pred(*args))

    for action in actions:
        print("Taking action: ", action)
        next_obs, rew, episode_done, _  = env.step(action)
        positive_effects = {e for e in next_obs.literals - obs.literals}
        negative_effects = {Anti(ne) for ne in obs.literals - next_obs.literals}
        effects =  positive_effects | negative_effects
        print('Effects:\n', effects, '\n\n')
        obs = next_obs

        if len(effects) == 0:
            obs_literals = set()
            for lit in obs.literals:
                if lit.predicate.name not in ('different', 'name-less-than'):
                    obs_literals.add(lit)
            print(f"State: ")
            pprint(sorted(obs_literals))
            raise Exception(f"Got unexpected null effect for action {action}")

        if episode_done:
            print("Reached goal! Reward: ", rew)
            input()
