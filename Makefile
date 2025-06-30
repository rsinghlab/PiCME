# Define linting-related packages and files
LINT_PACKAGES=black black-jupyter blacken-docs 
PYTHON_FILES=$(find . -type f -name '*.py')
IPYNB_FILES=$(find . -type f -name '*.ipynb')

# Enter your VENV here!
VENV="<enter your venv bin/activate path here>"

lint:
	@echo "Running linting on all Python files..."
	black .
	isort .

setup-lint-conda:
	@echo "Installing linting packages..."
	conda install $(LINT_PACKAGES)

setup-lint-venv:
	@echo "Installing linting packages..."
	pip install $(LINT_PACKAGES)

pretrain:
	sbatch model/pretrain.sh $(VENV)

finetune:
	sbatch model/finetune.sh $(VENV)

evaluate:
	sbatch model/evaluate.sh $(VENV)

sweep-tune-evaluate:
	sbatch model/sweep_tune_evaluate.sh $(VENV)

permission:
	chmod -R 777 .

.PHONY: lint setup-lint-conda setup-lint-venv finetune evaluate evaluate permission
