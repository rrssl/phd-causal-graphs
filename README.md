# Computational Design and Optimization of Rube Goldberg Machines

This repository contains the code used in the following peer-reviewed publication:
- Roussel, R., Cani, M.-P., Léon, J.-C., & Mitra, N. J. (2019). Designing chain reaction contraptions from causal graphs. ACM Transactions on Graphics, 38(4), 43:1-43:14. https://doi.org/10.1145/3306346.3322977


## Installation
Requires Python 3.7+ and `pip`.

Ubuntu:
```
sudo apt install openscad libgeos-dev libcairo2-dev libffi-dev python3-tk
pip3 install -r requirements.txt
```
macOS:
```
brew install cairo geos libffi
brew cask install openscad
pip install -r requirements.txt
```

## Contents
- `blender/` Scripts used to export animations to Blender
- `core/` Core package; contains all the algorithms
- `demos/` Demos to play with
- `gui/` Graphical modules
- `scenarios/` Config files of the scenarios presented in the paper

## Usage
```
cd demos/
python3 app_designer.py
python3 create_random_instance ../scenarios/ballrun.py
python3 import_scenario.py ../scenarios/ballrun.py 1
python3 optimize_robustness.py ../scenarios/ballrun.py
python3 plot_local_robustness.py ../scenarios/ballrun.py 10000 4
```

NB: some parameters must be specified directly in the scripts.
