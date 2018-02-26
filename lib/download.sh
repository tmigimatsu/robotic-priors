# cd NatNetLinux.git
# git submodule update --init
# cd ..

# TODO: Place inside sai2-common
curl -L https://bitbucket.org/eigen/eigen/get/3.3.4.zip -o eigen-3.3.4.zip
unzip eigen-3.3.4.zip
mv eigen-eigen-* eigen

# TODO: Place inside sai2-common
curl -L https://bitbucket.org/rbdl/rbdl/get/v2.5.0.zip -o rbdl-2.5.0.zip
unzip rbdl-2.5.0.zip
mv rbdl-rbdl-* rbdl
cd rbdl
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DRBDL_BUILD_ADDON_URDFREADER=ON -DRBDL_USE_ROS_URDF_LIBRARY=OFF ..
make -j4
cd ../..

cd chai3d.git
git submodule update --init
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
cd ../..

# cd sai2-common.git
# # git submodule update --init --recursive
# mkdir -p build_rel
# cd build_rel
# cmake -DCMAKE_BUILD_TYPE=Release ..
# make -j4
# cd ../..

# cd sai2-simulation.git
# # git submodule update --init --recursive
# mkdir -p build_rel
# cd build_rel
# cmake -DCMAKE_BUILD_TYPE=Release ..
# make -j4
# cd ../..

