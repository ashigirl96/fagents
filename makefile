PY=python

up:
	python setup.py install
watch: up
	watchmedo shell-command --patterns="**/*.py" --command=up
