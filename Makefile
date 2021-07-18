
run_isort:
	isort -q .

run_yapf:
	yapf -i -r .

run_autoflake:
	autoflake -i -r --remove-all-unused-imports --ignore-init-module-imports src/.

clean_repo: run_yapf  run_isort  run_autoflake
