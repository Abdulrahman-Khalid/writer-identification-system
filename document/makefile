all: report.pdf

%.pdf: %.tex
	pdflatex $^

clean:
	-rm *.aux *.log *.synctex.gz *.fls *.out *.fdb_latexmk
