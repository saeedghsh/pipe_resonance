.PHONY: create-env
create-env:
	@echo "\n=== Creating conda virtual environment (using mamba) ==="
	mamba env create -f environment/environment.yaml

# syncs with environment.yaml
# does not update existing environment even if it is outdated
# ignores lock file if it exists
.PHONY: update-env
update-env:
	@echo "\n=== Updating conda virtual environment (using mamba) ==="
	mamba env update -f environment/environment.yaml

# lock is full reproducibility, all packages & deps, cross-platform, CI/CD-friendly
# Install lock using: conda-lock install --name YOURENV environment-lock.yaml
.PHONY: lock-env
lock-env:
	@echo "\n=== Locking conda virtual environment ==="
	conda-lock lock --mamba -f environment/environment.yaml --lockfile environment/environment-lock.yaml

# pin is human-readable, top-level packages only, less reproducible, single-platform
.PHONY: pin-env
pin-env:
	@echo "\n=== Pinning conda virtual environment ==="
	mamba env export --from-history > environment/environment-pinned.yaml

# Checks which package in the currently activated conda env is outdated
.PHONY: outdated-env
outdated-env:
	@echo "\n=== Checking for outdated packages in conda virtual environment ==="
	mamba update --all --dry-run
