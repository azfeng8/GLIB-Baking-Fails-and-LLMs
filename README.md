# Installation
Instructions for running:
* Use Python 3.5 or higher, e.g. with a virtual environment.
* Download Python dependencies: `pip install -r requirements.txt`.
* Download the Fast-Forward (FF) planner to any location on your computer.
-> Linux: https://fai.cs.uni-saarland.de/hoffmann/ff/FF-v2.3.tgz
-> Mac: https://github.com/ronuchit/FF
* From the FF directory you just created, run `make` to build FF, producing the executable `ff`.
* Create an environment variable "FF_PATH" pointing to this `ff` executable.
* Follow the steps under the 'Setup OpenAI key' section below.
* Back in the GLIB directory, you can now run `python main.py`.

## Setup OpenAI key

Set OPEN_API_KEY as an environment variable with the API key.

Put in ~/.bashrc:

```
export OPEN_API_KEY="INSERT API KEY"
```

## settings.py

### LLMConfig
In LLMConfig, provide a path for caching LLM queries under variable `cache_dir`.

### AgentConfig
In AgentConfig, select the learning method:

To run warm-start method, set `learning_name`="LLM+LNDR"

To run iterative method, set `learning_name`="LLMIterative+ZPK" and uncomment  "LLM+GLIB_L2" and/or "LLM+GLIB_G1" in `curiosity_modules_to_run`.

After changing the settings, run `python main.py`.