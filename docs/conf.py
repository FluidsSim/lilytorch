# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup ---------------------------------------------------------------
# Add project root so autodoc can find ``lilytorch``
sys.path.insert(0, os.path.abspath(".."))

# -- Project information ------------------------------------------------------
project = "LilyTorch"
copyright = "2024–2026, Andrea Ferrario"
author = "Andrea Ferrario"
release = "0.1.0"

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",        # Google / NumPy-style docstrings
    "sphinx.ext.mathjax",         # LaTeX math rendering
    "sphinx.ext.viewcode",        # [source] links to highlighted source
    "sphinx.ext.intersphinx",     # cross-ref NumPy / PyTorch docs
    "sphinx.ext.todo",
    # "sphinx_copybutton",          # copy button on code blocks
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc settings ---------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autosummary_generate = True

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- Options for HTML output --------------------------------------------------
html_theme = "furo"
html_title = "LilyTorch Documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2962FF",
        "color-brand-content": "#2962FF",
    },
    "dark_css_variables": {
        "color-brand-primary": "#82B1FF",
        "color-brand-content": "#82B1FF",
    },
    "navigation_with_keys": True,
    "sidebar_hide_name": False,
    "top_of_page_button": "edit",
}

# -- Intersphinx mapping ------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- Math support -------------------------------------------------------------
mathjax3_config = {
    "tex": {
        "macros": {
            "vb": [r"\mathbf{#1}", 1],      # bold vector shortcut
            "pd": [r"\partial", 0],
            "dd": [r"\mathrm{d}", 0],
        }
    }
}

# -- Todo extension -----------------------------------------------------------
todo_include_todos = True
