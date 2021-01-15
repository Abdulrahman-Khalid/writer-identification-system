#!/bin/bash
DIRECTORY=~/.pypy3
VENV_DIRECTORY=~/.team1
VERSION=pypy3.6-v7.3.0-linux64

# step 1: sudo apt install virtualenv;
# step 3: pip3 install -r requirements.txt

# Download (or use existing) pypy3
if [ -d "$DIRECTORY" ]; then
    echo "Skipping PyPy download, already exists"
else
    echo "Downloading PyPy to $DIRECTORY"
    # Download & extract to DIRECTORY
    wget https://techoverflow.net/downloads/${VERSION}.tar.bz2 -O /tmp/${VERSION}.tar.bz2
    bash -c "cd /tmp && tar xjvf ${VERSION}.tar.bz2"
    mv /tmp/${VERSION} $DIRECTORY
    rm /tmp/${VERSION}.tar.bz2
fi

# Create virtualenv
if [ -d "$VENV_DIRECTORY" ]; then
    echo "Skipping to create pypy3 virtualenv, already exists"
else
    echo "Creating PyPy virtual environment in $VENV_DIRECTORY"
    virtualenv -p ${DIRECTORY}/bin/pypy3 ${VENV_DIRECTORY}
fi

# Create "vpypy" shortcut
set -x
team1
result="$?"
set +x
if [ "$result" -ne 127 ]; then
    echo "Skipping to create team1 shortcut, already exists in current shell"
else
    echo "Creating bash/zsh shortcut 'team1'"
    if [ -f ~/.bashrc ]; then
        echo -e "\n# TechOverflow PyPy installer\nalias team1='source ${VENV_DIRECTORY}/bin/activate'\n" >> ~/.bashrc
    fi
    if [ -f ~/.zshrc ]; then
        echo -e "\n# TechOverflow PyPy installer\nalias team1='source ${VENV_DIRECTORY}/bin/activate'\n" >> ~/.zshrc
    fi
    # Activate shortcut in current shell (but do not automatically activate virtual environment)
    alias team1='source ${VENV_DIRECTORY}/bin/activate'
fi

echo -e "\n\nPyPy installation finished. Restart your shell, then run 'team1' to activate the virtual environment"
