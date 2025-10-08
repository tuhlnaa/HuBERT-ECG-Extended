#!/bin/bash

# [Linux]
# chmod +x ./script/Install_dependencies.sh

echo "Update conda itself"
conda update -n base -c defaults conda -y

# Install gcc and gxx using conda
echo "Installing gcc..."
conda install -c conda-forge gcc -y

echo "Installing gxx..."
conda install -c conda-forge gxx -y

# Install libsndfile (audio processing)
echo "Installing libsndfile..."
conda install -c conda-forge libsndfile -y

while read line; do
  # Skip empty lines and comments
  if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
    echo "Installing $line..."
    pip install $line
  fi
done < requirements.txt