# Build everything:  make          (figures, then notes.pdf and exercises.pdf)
#        just PDFs:  make pdf
#        the code:   make test     /  make notebooks
#   solutions PDF:  make solutions SOLUTIONS=<path to this course in the private repo>
LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error
PDFS    = notes exercises
# The course id is the first part of the folder name (ieor222-..., mfe230d-...).
COURSE    ?= $(firstword $(subst -, ,$(notdir $(CURDIR))))
SOLUTIONS ?= ../teaching-solutions/$(COURSE)

.PHONY: all figures pdf $(PDFS) solutions test notebooks clean

all: figures pdf

figures:
	python code/figures/make_figures.py

# exercises and notes reference each other, so build them in turn.
pdf:
	cd latex && $(LATEXMK) exercises.tex && $(LATEXMK) notes.tex && $(LATEXMK) exercises.tex

$(PDFS):
	cd latex && $(LATEXMK) $@.tex

# Private: copies the solutions in, builds, and leaves them git-ignored.
solutions:
	@test -d "$(SOLUTIONS)" || { echo "Solutions folder not found: $(SOLUTIONS) (they live in the private repo)"; exit 1; }
	rm -rf latex/solutions && cp -R "$(SOLUTIONS)" latex/solutions
	cd latex && $(LATEXMK) notes.tex && $(LATEXMK) solutions.tex

test:
	pytest -q code/tests

notebooks:
	jupyter nbconvert --to notebook --execute --inplace code/notebooks/*.ipynb

clean:
	cd latex && latexmk -C notes.tex exercises.tex solutions.tex && rm -rf solutions
