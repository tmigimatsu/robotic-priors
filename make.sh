set -e

mkdir -p build
cd build
cmake ..
make -j4
cd ..

# Insert helper scripts into bin directory
cd bin

# Make script
cat <<EOF > make.sh
cd ..
mkdir -p build
cd build
cmake ..
make -j4
cd ../bin
EOF
chmod +x make.sh

# Run generic controller script
cat <<EOF > visualizer.sh
# /opt/VirtualGL/bin/vglrun ./sai2_env ../resources/gym.urdf ../resources/kuka_iiwa_gym.urdf kuka_iiwa
./sai2_env ../resources/gym.urdf ../resources/kuka_iiwa_gym.urdf kuka_iiwa
EOF
chmod +x visualizer.sh

cd ..
