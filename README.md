# Installation
Instructions for running:
* Use Python 3.5 or higher, e.g. with a virtual environment.
* Download Python dependencies: `pip install -r requirements.txt`.
* Download the Fast-Forward (FF) planner to any location on your computer.
-> Linux: https://fai.cs.uni-saarland.de/hoffmann/ff/FF-v2.3.tgz
-> Mac: https://github.com/ronuchit/FF
* From the FF directory you just created, run `make` to build FF, producing the executable `ff`.
* Create an environment variable "FF_PATH" pointing to this `ff` executable.
* Download the FastDownward planner repo. https://github.com/aibasel/downward, run `python build.py` in the Github repo.
* From the repo, create an environment variable "FD_PATH" pointing to the `fast-downward.py` file in that Github repo.
* Follow the steps under the 'Setup OpenAI key' section below.
* Back in the GLIB directory, you can now run `python main.py`.
# Steps to run LLM methods

## Setup OpenAI key

Set OPENAI_API_KEY as an environment variable with the API key.

Put in ~/.bashrc:

```
export OPENAI_API_KEY="INSERT API KEY"
```

Run `python main.py ...`.

## Required Arguments:

`--domains : str, +` :

list of PDDLGym domains.

`--curiosity_methods : str, +` :

list of curiosity methods. See settings.py for the complete list in AgentConfig.curiosity_methods_to_run.

`--learning_name : str` :

name of the learning method. See settings.py for the complete list in AgentConfig.learning_name. 

`--start_seed : int` :

starting seed number of the first seed. The following seeds increment from here.

`--num_seeds : int` :

number of seeds to run.


### Example Command:

```
python main.py --domains Baking  --curiosity_methods LLM+GLIB_G1  --learning_name LLMIterative+ZPK  --start_seed 40 --num_seeds 1
```
