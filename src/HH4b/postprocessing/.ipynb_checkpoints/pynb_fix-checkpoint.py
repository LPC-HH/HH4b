import nbformat

# Read the notebook
with open("DataMCPlots.ipynb", "r", encoding="utf-8") as f:
    notebook = nbformat.read(f, as_version=4)

# Write the notebook back out
with open("DataMCPlots_fixed.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(notebook, f)