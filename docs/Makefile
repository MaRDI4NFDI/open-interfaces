# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
# -W turns warnings into errors
# --keep-going lets sphinx process the full build instead of aborting on first warning
#   this is esp. necessary for myst-nb code-cells that otherwise show no visible code error
SPHINXOPTS    =
# serial build is necessary until sphinx-autoapi supports parallel
SPHINXBUILD   = sphinx-build -j 1 -v .
BUILDDIR      = _build

# Internal variables.
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(SPHINXOPTS)
# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(SPHINXOPTS) .

.PHONY: clean html



html:
	[ -d $(BUILDDIR)/cmake-build-debug ] || cmake -S .. -B $(BUILDDIR)/cmake-build-debug
	doxygen
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

clean:
	-rm -rf $(BUILDDIR)/*
	-rm -f source/generated/*.rst
