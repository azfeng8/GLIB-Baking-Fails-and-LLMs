
# Modify settings.json and settings.py to run `python main.py`

Pick one of four modes to write in settings.json.

```
{
    # "run": true,
    # "runSave": true,
    # "loadRun": {
        # "explorer": "GLIB_G1",
        # "domain": "Baking"
        # "path": ""
        # "experiment_no": 0
    # }
    "loadSave": {
        "explorer": "GLIB_L2",
        "domain": "Baking"
        "path": "path_to_settings.py"
        "experiment_no": 0
    },

}

```

### 4 Modes:

<b>"runSave"</b>: run from this directory's settings.py and save outputs as new experiments

<b>"run" </b>: run from this directory's settings.py

<b>"loadSave" </b>: load the settings from the path specified, running the experiment specified by the number. Also save the replay's output as a new experiment.

<b>"loadRun" </b>: load the settings from the path specified, running the experiment specified by the number. 

Loading will only load one "experiment": one explorer on one domain for one run. NOTE that the explorer and domain name must be specified in the settings.py file to be valid.

### 'loadSave' and 'loadRun' fields:

<b>path</b> (str): path to the settings.py to load from

<b>explorer</b> (str): name of explorer to run, one of the names in the AgentConfig of the python settings file.

<b>domain</b> (str): name of domain to run on, must be a valid PDDLGym environment name

<b>experiment_id</b> (str): ID of the experiment (the suffix name of the folder in the folder structure in the next section)


# Folder structure of experiments log (WIP)

```
{domain_name}
    {explorer_name}
        experiment_{ID}/
            settings.py
            explorer_summary.json
            results.pkl

            iter_{#}/
                explorer.json
                after.pddl
                ndrs.txt
                successes.txt
```

## experiment-level logs:

### explorer_summary.json:

"plans": a list of iteration #s where plans were found

"actions": a list of actions taken on each iteration

"random_or_not": a list with 1s,0s. 1 if the action taken that iteration was result of a babble and plan, or 0 if used fallback to a random action.


## iteration-level logs:

### after.pddl

Domain file with updated operators after taking the action in that iteration. This filename exists in the iterations when the operators were updated.

Operators should have the maximum likelihood (determinized) effects from ndrs.txt.

### ndrs.txt

Human-readable file to see the noisy deictic rule set after taking the action and learning in that iteration. This filename exists in the iterations when operators were updated.

### successes.txt

Array of 1's 0's, ordered in the PDDLGym test problem indices. 1 if reached the goal for the problem using the operators after the action in this iteration, else 0.

### explorer.json

"babbled": list of parseable goal,action strings that were babbled

"action": string parseable grounded action taken

"plan_found":
        
        {
            "goal": the lifted goal babbled
            "action": the lifted action babbled
            "plan": the plan found (list of grounded actions)
        }

"empty_plan_so_grounded_action": planner returned an empty plan

        {
            "goal": the lifted goal babbled
            "action": the lifted action babbled
            "plan": the plan found (list of grounded actions, should be empty)
        }

"found_no_plans_so_random_action": True if random action is taken because no plans were found within the budget of sampling tries

"empty_plan_so_random_action": True if random action is taken because the planner returned an empty plan and the babbled lifted action is not able to be grounded in the current state

"following_plan": True if following a previously found plan

"action_after_plan": True if executing the babbled action after following a plan


## GLIB: Efficient Exploration for Relational Model-Based Reinforcement Learning via Goal-Literal Babbling

Link to paper: https://arxiv.org/abs/2001.08299
