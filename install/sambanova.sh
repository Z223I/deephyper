#!/bin/bash -x

module load postgresql
module load miniconda-3
conda create -p dh-sambanova python=3.8 -y
conda activate dh-sambanova/
conda install gxx_linux-64 gcc_linux-64 -y
# DeepHyper + Analytics Tools (Parsing logs, Plots, Notebooks)
pip install deephyper[analytics,balsam]
conda install tensorflow -c intel -y


# Checking existence of "bashrc_sambanova"
BASHRC_SAMBANOVA=~/.bashrc_sambanova

read -r -d '' NEW_BASHRC_CONTENT <<- EOM
# Added by DeepHyper
if [[ $(echo '$HOSTNAME') = *"sm"* ]]; then
    source ~/.bashrc_sambanova
fi
EOM

if test -f "$BASHRC_SAMBANOVA"; then
    echo "$BASHRC_SAMBANOVA exists."
else
    echo "$BASHRC_SAMBANOVA does not exists."
    echo "Adding new lines to ~/.bashrc"
    echo "$NEW_BASHRC_CONTENT" >> ~/.bashrc
fi


read -r -d '' NEW_BASHRC_SAMBANOVA_CONTENT <<- EOM
# Added by DeepHyper
module load postgresql
module load miniconda-3
EOM

echo "Adding new lines to $BASHRC_SAMBANOVA"
echo "$NEW_BASHRC_SAMBANOVA_CONTENT" >> $BASHRC_SAMBANOVA
