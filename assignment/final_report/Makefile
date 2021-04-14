FILES = input/introduction.md \
	input/background.md \
	input/methods.md \
	input/results.md \
	input/conclusion.md

OUTPUT = build

FLAGS = -f markdown+smart -s --metadata-file=metadata.yaml \
	--standalone --self-contained --listings

FLAGS_PDF = --template=template.latex \
	    --pdf-engine=tectonic \

all: pdf

latex:
	pandoc -o $(OUTPUT)/paper.latex $(FLAGS) $(FLAGS_PDF) $(FILES)

pdf:
	pandoc -o $(OUTPUT)/paper.pdf $(FLAGS) $(FLAGS_PDF) $(FILES)

clean:
	rm build/*
