
# Modify settings.json

{
    # "runSave": "settings.py"
    # "run": "settings.py"
    # "loadRun": {
        # "explorer": "GLIB_G1",
        # "domain": "Baking"
        # "path": ""
    # }
    "loadSave": {
        "explorer": "GLIB_L2",
        "domain": "Baking"
        "path": "path_to_settings.py"
    },

}

Modes:

"runSave"
"run"
"loadSave"
"loadRun"

Loading will only load one "experiment": one explorer on one domain for one run.

Load fields:
    path: path to the settings.py to load from
    explorer: name of explorer to run 
    domain: name of domain to run on
    # NOTE that the explorer and domain name must be specified in the settings.py file to be valid.



GLIB: Efficient Exploration for Relational Model-Based Reinforcement Learning via Goal-Literal Babbling

Link to paper: https://arxiv.org/abs/2001.08299
