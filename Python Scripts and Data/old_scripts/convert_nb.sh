#!/usr/bin/env bash
# convert_nb.sh ─ Convert a Jupyter notebook to a .py file
# Usage:  ./convert_nb.sh  [optional-.ipynb-file]

set -e  # exit on first error

# If you pass a notebook name, use it; otherwise default
NB_FILE="${1:-Crack_Arrest_Notebook.ipynb}"

# Make sure the file exists
if [[ ! -f "$NB_FILE" ]]; then
  echo "Error: '$NB_FILE' not found"; exit 1
fi

# Strip the .ipynb suffix for the output name
BASE_NAME="${NB_FILE%.ipynb}"

# Run the conversion (requires jupyter & nbconvert)
jupyter nbconvert --to script --output "$BASE_NAME" "$NB_FILE"
echo "✓ Converted to ${BASE_NAME}.py"
