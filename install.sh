set -e

# ---------------------------------------
# Install precompiled 3rd party libraries
# ---------------------------------------

if [[ "$OSTYPE" == "linux-gnu" ]]; then
	sudo apt-get install curl cmake libeigen3-dev libtinyxml2-dev libjsoncpp-dev libhiredis-dev libglfw3-dev xorg-dev freeglut3-dev libasound2-dev libusb-1.0-0-dev redis-server
	# Install gcc 5 for Ubuntu 14.04:
	# sudo add-apt-repository ppa:ubuntu-toolchain-r/test
	# sudo apt-get update
	# sudo apt-get install gcc-5 g++-5
	# sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 60 --slave /usr/bin/g++ g++ /usr/bin/g++-5
elif [[ "$OSTYPE" == "darwin"* ]]; then
	brew install cmake eigen redis hiredis tinyxml2 jsoncpp glfw3
fi

mkdir -p lib
cd lib

./download.sh

cd ..
