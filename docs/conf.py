# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Adiciona o raiz do repositório ao path para que autodoc encontre os módulos
sys.path.insert(0, os.path.abspath('..'))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = 'DDSP-Sonar'
copyright = '2026, Marlon Jovenil de Souza'
author = 'Marlon Jovenil de Souza'
release = '0.1.0'

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # Gera docs a partir das docstrings
    'sphinx.ext.napoleon',      # Suporte a Google-style e NumPy-style docstrings
    'sphinx.ext.viewcode',      # Adiciona link "Ver código-fonte" em cada símbolo
    'sphinx.ext.intersphinx',   # Links para docs externas (PyTorch, Python)
    'sphinx.ext.mathjax',       # Renderiza equações LaTeX no HTML
    'sphinx.ext.autosummary',   # Tabelas de resumo automáticas
]

# Napoleon: configurar para Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc: comportamento padrão
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',   # Mantém a ordem do código-fonte
    'special-members': '__init__',
    'undoc-members': False,       # Só documenta o que tem docstring
    'show-inheritance': True,
}
autodoc_typehints = 'description'  # Tipos aparecem na descrição, não na assinatura

# Autosummary: gerar stub .rst automaticamente
autosummary_generate = True

# Links para documentação externa
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

# Arquivos e padrões a ignorar
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

templates_path = ['_templates']

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'   # Read the Docs theme — padrão acadêmico
html_static_path = ['_static']

html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
}

# Título na barra lateral
html_title = 'DDSP-Sonar — Documentação'
html_short_title = 'DDSP-Sonar'
