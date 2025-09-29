source ~/emsdk/emsdk_env.sh

emcmake cmake -B build -G Ninja

cmake --build build --parallel

cmake --install build
