PY ?= python

.PHONY: help install test bellman dp mc td fa fqi pg ac all clean

help:
	@echo "make install   install the pinned dependencies"
	@echo "make test      run the test suite"
	@echo "make all       run every chapter (several hours)"
	@echo "make bellman | dp | mc | td | fa | fqi | pg | ac   run one chapter"

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests -q

bellman:
	cd 01_bellman_equations && $(PY) bellman_equations.py

dp:
	cd 02_dynamic_programming && $(PY) dynamic_programming.py

mc:
	cd 03_monte_carlo && $(PY) monte_carlo_control.py

td:
	cd 04_temporal_difference && $(PY) td_control.py

fa:
	cd 05_function_approximation && $(PY) features_demo.py \
	  && $(PY) regression_1d.py \
	  && $(PY) make_dataset.py \
	  && $(PY) semi_gradient_td.py

fqi:
	cd 06_fitted_q_iteration && $(PY) fitted_q_iteration.py

pg:
	cd 07_policy_gradient && $(PY) reinforce.py

ac:
	cd 08_actor_critic && $(PY) actor_critic.py

all: test bellman dp mc td fa fqi pg ac

clean:
	rm -rf */figures/*.png */results/*.csv */results/*.md \
	       05_function_approximation/data/*.npz
	find . -name __pycache__ -type d -exec rm -rf {} +
