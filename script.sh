sudo apt install build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev pypy3-dev libtbb2 libtbb-dev libdc1394-22-dev;

mkdir ~/opencv_build && cd ~/opencv_build;
git clone https://github.com/opencv/opencv.git;
git clone https://github.com/opencv/opencv_contrib.git;

cd ~/opencv_build/opencv;
mkdir build && cd build;

cmake -D PYTHON3_LIBRARY=~/.pypy3.6-7.3.0/lib/libpypy-c.so -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules;

make -j8;

sudo make install;
