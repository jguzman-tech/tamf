# Targeted Active-Measurement Framework (TAMF)

This repository contains code that works to implement a TAMF. TAMF is a method
to efficiently deploy resources in order to measure the level of Wireless
Broadband of a region. This is a concern especially in rural tribal
regions. This framework will assign a utility metric to 1km blocks of a region
based on the value of coverage (VoC) and existing evidence of coverage (EoC) of
that grid block.

# Setup
Core functionality was wrapped into a python pip module. Below is how you would
install.
## git clone
	git clone https://github.com/jguzman-tech/tamf.git
## enter directory
	cd ./tamf
## install via pip
	python -m pip install --editable .
There may be more updates in the near future, using the --editable option lets
you edit the files and still use pip. Issues can arise otherwise if you make
significant edits to a pip package without re-installing.
# Usage
TBD
