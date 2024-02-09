# Steps to run LLM methods

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