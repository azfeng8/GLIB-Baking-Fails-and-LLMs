# Logging and loading

This branch contains utilities for logging and loading.

## Purposes
Wrote some logging and loading utilities. Purposes are:

1. To analyze the methods and implementations more clearly.
2. Replay experiments in a way to be able to compare outputs for isolating the effects of the learner, planner, and explorer. (holding all other variables (modules) constant)
3. Load transition model: for LLM warm-starting.

## Instructions

Modify settings.json to collect,analyze,replay (or not) experiments. To modify each experiment's details, use settings.py.

Pick one of four modes to write in settings.json.

```
{
    # "run": true,
    # "runSave": true,
     "loadRun": {
        "experiment_path": "", # defines the domain, explorer, and seed
        "operators": 39, # initializes the learned model and NDRs from iteration 39
    #    "actions": 39, # iteration to start executing actions. most of time will be same iter # as the operators
    #    "goalactions": 39,  #iteration to start babbling goalactions. most of time will be same iter as the operators
    #    "goalactions_random": 39,
     }
    # "loadSave": {
    #    "experiment_path": "",
    #    "operators": 39,
    #    "replay_experiment": true # initializes LNDR at iteration 0, and executes logged actions. to check for randomness in LNDR implementation.
    },
}

```

### 4 Modes:

<b>"runSave"</b>: run from this directory's settings.py and save outputs as new experiments

<b>"run" </b>: run from this directory's settings.py

<b>"loadSave" </b>: load the settings from the path specified, running the experiment specified by the number. Also save the replay's output as a new experiment.

<b>"loadRun" </b>: load the settings from the path specified, running the experiment specified by the number. 


### 'loadSave' and 'loadRun' fields:

<b>path</b> (str): path to the experiment folder

<b> operators</b> (int): iteration number of the PDDL and NDRs to load.

<b> actions </b> (int): iteration number of the actions to load. Specify at most one of ["goalactions", "actions", "goalactions_random"].

<b> goalactions </b> (int):  iteration number of the babbled (goal, action) pairs to load, with the same action fallbacks. Specify at most one of ["goalactions", "actions", "goalactions_random"].

<b> goalactions_random> </b> (int): iteration number of the babbled (goal, action) pairs to load, with random sampled action fallbacks (should be same b/c same seed?). Specify at most one of ["goalactions", "actions", "goalactions_random"].


### 'loadSave' field:

<b> replay_experiment </b> (bool): if this field is true, the previous 3 fields are nullified. Makes sense to use for testing reproducibility / randomness in implementation.

# Folder structure of experiments log

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
                state.txt
```

## experiment-level logs:

### explorer_summary.json:

"plans": a list of iteration #s where plans were found

"actions": a list of actions taken on each iteration

"planned_action_or_not": a list with 1s,0s. 1 if the action taken that iteration was result of a babble and plan, or 0 if used fallback to a random action.

"random_action_no_change": a list of iteration # where random action was taken and no change in the state was observed.


## iteration-level logs:

### after.pddl

Domain file with updated operators after taking the action in that iteration. This filename exists in the iterations when the operators were updated.

Operators should have the maximum likelihood (determinized) effects from ndrs.txt.

### ndrs.txt

Human-readable file to see the noisy deictic rule set after taking the action and learning in that iteration. This filename exists in the iterations when operators were updated.

### successes.txt

Array of 1's 0's, ordered in the PDDLGym test problem indices. 1 if reached the goal for the problem using the operators after the action in this iteration, else 0.

### State

State to be given as a prompt to the LLM.

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

# Scripts to analyze experiment data

### Identify checkpoints

### Get % random actions taken, and % of random actions that have null effect



# Plotting success rate and variational distance

To plot and compare data from curiosity modules from multiple experiments, put the .pkl results from the experiment logs into the `results/` folder, and run with `AgentConfig.cached_results_to_load` field with the explorer.

```
results/
    domain_name/
        learning_name/
            curiosity_name/
                *.pkl
```


## GLIB: Efficient Exploration for Relational Model-Based Reinforcement Learning via Goal-Literal Babbling

Link to paper: https://arxiv.org/abs/2001.08299
