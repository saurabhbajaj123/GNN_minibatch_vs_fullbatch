apt install wget
apt install git
apt install libgoogle-glog-dev
apt install libgtest-dev
cd /usr/src/gtest
mkdir build && cd build
cmake ..
make 
cp libgtest*.a /usr/local/lib
apt install libgflags-dev
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz
tar zxvf openmpi-4.0.1.tar.gz
cd openmpi-4.0.1
./configure --prefix=/opt/openmpi_install
make install
wget https://github.com/Kitware/CMake/releases/download/v3.15.1/cmake-3.15.1-Linux-x86_64.tar.gz
tar zxvf cmake-3.15.1-Linux-x86_64.tar.gz
apt install libzmq3-dev
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
tar zxvf metis-5.1.0.tar.gz
cd metis-5.1.0
make config prefix=/opt/metis_install shared=1
make
make install
cd /opt
mkdir -p nlohmann/nlohmann && cd nlohmann/nlohmann
wget https://github.com/nlohmann/json/releases/download/v3.7.3/json.hpp
apt install g++-5
export GCCL_CUDA=1     # Build with CUDA
export GCCL_GPU_TEST=1 # Build GPU test
export GCCL_MPI_TEST=1 # Build MPI test
export METIS_HOME=/opt/metis # METIS installation directory
export METIS_DLL=/opt/metis_install/lib/libmetis.so # METIS dll
export MPI_HOME=/opt/openmpi_install # MPI installation directory
export NLOHMANN_HOME=/opt # Nlohmann json installation home
export GCCL_HOME=/opt/gccl # GCCL installation home
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
export PATH=$PATH:/opt/cmake-3.15.1-Linux-x86_64/bin #cmake-3.15.1 installation home
export PATH=$PATH:/opt/openmpi_install/bin
mkdir build
cp scripts/build.sh build/
cd build
./build.sh
make -j
make test
make install