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