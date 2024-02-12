# Steps to run LLM methods

## Setup OpenAI key

Set OPEN_API_KEY as an environment variable with the API key.

Put in ~/.bashrc:

```
export OPEN_API_KEY="INSERT API KEY"
```

Run `python main.py ...`.

## Required Arguments:

--domains : str, + :

list of PDDLGym domains.

--curiosity_methods : str, +:

list of curiosity methods. See settings.py for the complete list in AgentConfig.curiosity_methods_to_run.

--learning_name : str:

name of the learning method. See settings.py for the complete list in AgentConfig.learning_name. 

--start_seed : int:

starting seed number of the first seed. The following seeds increment from here.

--num_seeds : int:

number of seeds to run.


### An example command is:

python main.py --domains Baking  --curiosity_methods LLM+GLIB_G1  --learning_name LLMIterative+ZPK  --start_seed 40 --num_seeds 1