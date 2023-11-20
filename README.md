
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
            final_operators.txt
            results.pkl

            iter_{#}/
                explorer.json
                learner.json
```

## experiment-level logs:

### explorer_summary.json:

"plans": a list of iteration #s where plans were found

"actions": a list of actions taken on each iteration

"babbled_or_not": a list with 1s,0s. 1 if the action taken that iteration was result of a babble and plan, or 0 if used fallback to a random action.


## iteration-level logs:

### learner.json
"operators_before":
"operators_after":
"operators_changed"

### explorer.json

"babbled": list of goal,action strings that were babbled

"action": string parseable grounded action taken

"plan_found": false, or:
        
        {
            "goal": the grounded goal babbled
            "plan": the plan found (list of grounded actions)
        }

"random_action": true if action taken is random, or false if following a plan found

# Verbosity levels

0

1

2

3

4

GLIB: Efficient Exploration for Relational Model-Based Reinforcement Learning via Goal-Literal Babbling

Link to paper: https://arxiv.org/abs/2001.08299
