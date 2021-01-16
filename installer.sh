#!/bin/bash
DIRECTORY=~/.pypy3
VENV_DIRECTORY=~/.team1
VERSION=pypy3.7-v7.3.3-linux64

# source ~/.bashrc
# team1
# step 1: sudo apt install virtualenv;
# sudo apt-get install pypy3-dev
# step 3: pypy3 -m pip install -r requirements.txt
# pypy3 -m ensurepip
# pypy3 -m pip install numpy
# pypy3 -m pip install scikit-build
# pypy3 -m pip install dlib
# step 4: CMAKE_ARGS="-D PYTHON3_LIBRARY=~/.pypy3.6-7.3.0/lib/libpypy-c.so" python3 setup.py bdist_wheel
# pypy3 -m pip install dist/**.wheel

# cmake -D PYTHON3_LIBRARY=~/.pypy3.6-7.3.0/lib/libpypy-c.so -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules
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
