# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MaRDI Open Interfaces"
copyright = "2023--2024 MaRDI Open Interfaces authors"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

numfig = True
math_numfig = True
numfig_secnum_depth = 2
math_eqref_format = "Eq. {number}"  # Space after . is non-breaking space!!!

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["build"]

myst_enable_extensions = ["dollarmath", "amsmath"]
myst_dmath_double_inline = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
