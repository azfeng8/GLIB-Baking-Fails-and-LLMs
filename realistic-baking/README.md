# Creating the new realistic Baking domain

This depends on many factors, such as:

- What predicate names are most suitable for the LLM to work
- How PDDLGym will simulate the environment
- If the operator learner can learn this large, complex domain
- How realistic the domain will be

To help prototype/develop the domain, I've created several scripts:

## create_ground_actions.py

Insert the list of skill predicates into the string at the top of the file, and then run it on a PDDL problem file to edit in place.

The PDDL problem file needs to have no ground actions in the (:init ) section, and have a comment called "all ground actions" in the section, below which the ground actions will be inserted.

The actions string is printed and needs to be copied to the domain file.

## create_different_preds_in_problem_and_domain_files.py

This file takes the current PDDL domain and problem files and adds the "different" and "name-less-than" predicates to the files. These predicates will make sure that the simulation only finds one possible
assignment in cases where it is ambiguous:

- For example, if the precondition specifies that there are two whole eggs in a bowl, then the assignment of variables ?x0: egg-0 ?x1: egg-2 and the assignment ?x0: egg-2 ?x1: egg-0 would both work, and cause pddlgym to fail. Using
'(name-less-than ?x0 ?x1)' rules out the second assignment, and will allow pddlgym to find only one assignment and cause it not to fail.

- For example, if the precondition specifies that there are two whole eggs in a bowl, then the assignment of variables ?x0: egg-0 ?x1: egg-2 and the assignment ?x0: egg-2 ?x1: egg-2 and the assignment ?x0: egg-0 ?x1: egg-0 would all work, and cause pddlgym to fail. Using '(different ?x0 ?x1)' rules out the second and third assignments, so that pddlygm only finds one assignment and won't fail.

## load_and_execute_plans.py

This file takes in hardcoded plans and executes them in pddlgym environment, printing out effects after each action, and notifying if the agent has reached the goal. This is to validate that the domain is working correctly.