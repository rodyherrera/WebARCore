source ~/emsdk/emsdk_env.sh

emcmake cmake -B build -G Ninja

cmake --build build --  arallel

cmake --install build
