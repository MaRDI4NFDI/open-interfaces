# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MaRDI Open Interfaces"
copyright = "2023--2025 MaRDI Open Interfaces authors"
release = "0.6.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

numfig = True
math_numfig = True
numfig_secnum_depth = 2
math_eqref_format = "Eq. {number}"  # Space after . is non-breaking space!!!
numfig_format = {
    "figure": "Figure %s:",
}

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.imgconverter",
    "autoapi.extension",  # Generate API documentation from docstrings
    "sphinx.ext.napoleon",  # Parse NumPy and Google style docstrings
    "breathe",  # Generate API documentation from Doxygen XML
]

templates_path = ["_templates"]
exclude_patterns = ["build"]

myst_enable_extensions = ["dollarmath", "amsmath"]
myst_dmath_double_inline = True

# Convert PDFs to PNGs such that images look good on HiDPI displays.
image_converter_args = ["-density", "600"]

# -- Options for sphinx-autoapi extension -------------------------------------------
autoapi_dirs = ["../../lang_python/oif_interfaces/openinterfaces/interfaces"]
autoapi_options = ["show-inheritance", "members", "undoc-members"]
autoapi_add_toctree_entry = False
autoapi_keep_files = True
autoapi_root = "api/api-python"
autoapi_type = "python"

# -- Options for napoleon extension -------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- Options for breathe extension -------------------------------------------
breathe_projects = {"oif": "../build/doxygen/xml"}
breathe_default_project = "oif"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
