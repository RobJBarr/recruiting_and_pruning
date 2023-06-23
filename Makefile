setup: install
install: venv
	. venv/bin/activate && pip install -r requirements.txt

venv:
	test -d venv || python3 -m venv venv

clean:
	rm -rf venv
	rm -rf logs
	rm -rf __pycache__
	rm -f python_outputs/*.txt
	rm -f slurm_outputs/*.out
	rm -f hyperparameters/*.json