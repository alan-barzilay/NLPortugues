# globals

PYTHON_VERSION_FULL = $(shell python --version 2>&1)
PIP_VERSION_FULL = $(wordlist 1,2,$(shell pip --version 2>&1))

# custom targets

.PHONY: environment
## create environment
environment:
	pyenv install -s 3.7.2
	pyenv virtualenv 3.7.2 nlportugues
	pyenv local nlportugues

.PHONY: requirements
## install all requirements
requirements:
	pip install -Ur requirements.txt

.PHONY: flake-check
## check PEP-8 and other standards with flake8
flake-check:
	@echo ""
	@echo "\033[33mFlake 8 Standards\033[0m"
	@echo "\033[33m=================\033[0m"
	@echo ""
	@python -m flake8 && echo "\n\n\033[32mSuccess\033[0m\n" || (echo \
	"\n\n\033[31mFailure\033[0m\n\n\033[34mManually fix the offending \
	issues\033[0m\n" && exit 1)

.PHONY: black-check
## check Black code style
black-check:
	@echo ""
	@echo "\033[33mBlack Code Style\033[0m"
	@echo "\033[33m================\033[0m"
	@echo ""
	@python -m black --check --exclude="build/|buck-out/|dist/|_build/\
	|pip/|env/|\.pip/|\.git/|\.hg/|\.mypy_cache/|\.tox/|\.venv/" . \
	&& echo "\n\n\033[32mSuccess\033[0m\n" || (python -m black --diff \
	--exclude="build/|buck-out/|dist/|_build/|pip/|env/|\.pip/|\.git/|\
	\.hg/|\.mypy_cache/|\.tox/|\.venv/" . 2>&1 | grep -v -e reformatted -e done \
	&& echo "\n\033[31mFailure\033[0m\n\n\
	\033[34mRun \"\e[4mmake black\e[24m\" to apply style formatting to your code\
	\033[0m\n" && exit 1)

.PHONY: black
## apply the Black code style to code
black:
	black --exclude="build/|buck-out/|dist/|_build/|pip/|env/|\.pip/|\.git/|\.hg/|\.mypy_cache/|\.tox/|\.venv/" .

.PHONY: checks
## perform code standards and style checks
checks: flake-check black-check

.PHONY: profile
## show project environment profile
profile:
		@echo "PYTHON_VERSION: $(PYTHON_VERSION_FULL)"
		@echo "PIP_VERSION: $(PIP_VERSION_FULL)"

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
