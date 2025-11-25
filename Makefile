.PHONY: create-env
create-env:
	@echo "\n=== Creating conda virtual environment (using mamba) ==="
	mamba env create -f environment/environment.yaml

# syncs with environment.yaml - does not update existing environment even if it is outdated
.PHONY: update-env
update-env:
	@echo "\n=== Updating conda virtual environment (using mamba) ==="
	mamba env update -f environment/environment.yaml

.PHONY: formatter
formatter:
	@echo "\n=== Checking code formatting ==="
	@black --check .

.PHONY: linter
linter:
	@echo "\n=== Linting Python files (all) ==="
	@pylint $(shell git ls-files '*.py')

MYPY_OPTS = --install-types --non-interactive --explicit-package-bases --config-file=pyproject.toml --show-error-codes

.PHONY: type-check
type-check:
	@echo "\n=== Running type checks (all) ==="
	@mypy $(MYPY_OPTS) .

.PHONY: code-quality
code-quality:
	-@$(MAKE) formatter
	-@$(MAKE) type-check
	-@$(MAKE) linter
