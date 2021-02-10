FILES = input/introduction.md \
	input/background.md \
	input/methods.md \
	input/results.md \
	input/conclusion.md

METADATA = metadata.yml

OUTPUT = build

FLAGS = --bibliography=bibliography.bib \
		-s \
		-f markdown

FLAGS_PDF = --template=template.latex --pdf-engine=tectonic

all: pdf

latex:
	pandoc -o $(OUTPUT)/paper.latex $(FLAGS) $(FILES) $(METADATA)

pdf:
	pandoc -o $(OUTPUT)/paper.pdf $(FLAGS) $(FLAGS_PDF) $(FILES) $(METADATA)

clean:
	rm build/*
