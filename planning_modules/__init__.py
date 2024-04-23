from .ff import FastForwardPlanner
from .ffreplan import FFReplanner

def create_planning_module(planning_module_name, planning_operators, learned_operators, domain_name,
                           action_space, observation_space):
    if planning_module_name.lower() == "ff":
        return FastForwardPlanner(planning_operators, learned_operators, domain_name, action_space, 
            observation_space)
    if planning_module_name.lower() == "ffreplan":
        raise NotImplementedError("Learning and planning operators are not separated")
        return FFReplanner(planning_operators, domain_name, action_space, 
            observation_space)
    raise Exception("Unrecognized planning module '{}'".format(planning_module_name))
