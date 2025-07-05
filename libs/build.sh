#!/usr/bin/env bash

export CMAKE_POLICY_VERSION_MINIMUM=3.5
export CMAKE_POLICY_VERSION=3.5

set -euo pipefail

LIB_ROOT="$PWD"

EMSDK="${EMSDK:-$HOME/emsdk}"
EMSCRIPTEN_DIR="$EMSDK/upstream/emscripten"
EMSCRIPTEN_CMAKE_DIR="$EMSCRIPTEN_DIR/cmake/Modules/Platform/Emscripten.cmake"

# CMake version check
cmake_version=$(cmake -E compare_files <(echo) <(echo)) || true 
required_major=3 required_minor=5
read -r cmake_major cmake_minor _ <<<"$(cmake --version | awk 'NR==1{split($3,v,".");print v[1],v[2]}')"
if (( cmake_major<required_major || (cmake_major==required_major && cmake_minor<required_minor) )); then
  echo "✖ CMake >= 3.5 is required, but found ${cmake_major}.${cmake_minor}. Please update CMake and try again."
  exit 1
fi

# Accepts: DEFAULT | SIMD | THREADS
BUILD_TYPE="${1:-DEFAULT}"

case "$BUILD_TYPE" in
  SIMD)
    echo "⚙️  Building with SIMD enabled"
    INSTALL_DIR="$LIB_ROOT/build_simd"
    CXX_FLAGS="-O3 -std=c++23 -msimd128"
    C_FLAGS="-O3 -std=c23 -msimd128"
    CONF_OPENCV="--simd"
    ;;
  THREADS)
    echo "⚙️  Building with THREADS enabled"
    INSTALL_DIR="$LIB_ROOT/build_threads"
    CXX_FLAGS="-O3 -std=c++23 -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=4"
    C_FLAGS="-O3 -std=c23 -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=4"
    CONF_OPENCV="--threads"
    ;;
  *)
    echo "⚙️  Building with DEFAULT settings"
    INSTALL_DIR="$LIB_ROOT/build"
    CXX_FLAGS="-O3 -std=c++23"
    C_FLAGS="-O3 -std=c23"
    CONF_OPENCV=""
    ;;
esac

mkdir -p "$INSTALL_DIR"

build_OPENCV(){
  rm -rf "$INSTALL_DIR/opencv" "$LIB_ROOT/opencv/build"
  python "$LIB_ROOT/opencv/platforms/js/build_js.py" "$LIB_ROOT/opencv/build" \
         --build_wasm $CONF_OPENCV --emscripten_dir "$EMSCRIPTEN_DIR"
  cp -r "$LIB_ROOT/opencv/build" "$INSTALL_DIR/opencv"
}

build_EIGEN(){
  rm -rf "$INSTALL_DIR/eigen" "$LIB_ROOT/eigen/build"
  mkdir -p "$LIB_ROOT/eigen/build"
  pushd "$LIB_ROOT/eigen/build" >/dev/null
  emcmake cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_TOOLCHAIN_FILE="$EMSCRIPTEN_CMAKE_DIR" \
    -DCMAKE_C_FLAGS="$C_FLAGS" \
    -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/eigen" \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTING=OFF
  emmake make -j install
  popd >/dev/null
}

build_OBINDEX2(){
  rm -rf "$INSTALL_DIR/obindex2" "$LIB_ROOT/obindex2/build"
  mkdir -p "$LIB_ROOT/obindex2/build"
  pushd "$LIB_ROOT/obindex2/build" >/dev/null
  emcmake cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_TOOLCHAIN_FILE="$EMSCRIPTEN_CMAKE_DIR" \
    -DCMAKE_C_FLAGS="$C_FLAGS -s USE_BOOST_HEADERS=1" \
    -DCMAKE_CXX_FLAGS="$CXX_FLAGS -s USE_BOOST_HEADERS=1" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/obindex2" \
    -DBUILD_SHARED_LIBS=OFF \
    -DOpenCV_DIR="$LIB_ROOT/opencv/build"
  emmake make -j install
  popd >/dev/null
}

build_IBOW_LCD(){
  rm -rf "$INSTALL_DIR/ibow_lcd" "$LIB_ROOT/ibow_lcd/build"
  mkdir -p "$LIB_ROOT/ibow_lcd/build"
  pushd "$LIB_ROOT/ibow_lcd/build" >/dev/null
  emcmake cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_TOOLCHAIN_FILE="$EMSCRIPTEN_CMAKE_DIR" \
    -DCMAKE_C_FLAGS="$C_FLAGS -s USE_BOOST_HEADERS=1" \
    -DCMAKE_CXX_FLAGS="$CXX_FLAGS -s USE_BOOST_HEADERS=1" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/ibow_lcd" \
    -DBUILD_SHARED_LIBS=OFF \
    -DOpenCV_DIR="$LIB_ROOT/opencv/build"
  emmake make -j install
  popd >/dev/null
}

build_SOPHUS(){
  rm -rf "$INSTALL_DIR/Sophus" "$LIB_ROOT/Sophus/build"
  mkdir -p "$LIB_ROOT/Sophus/build"
  pushd "$LIB_ROOT/Sophus/build" >/dev/null
  emcmake cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_TOOLCHAIN_FILE="$EMSCRIPTEN_CMAKE_DIR" \
    -DCMAKE_C_FLAGS="$C_FLAGS" \
    -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/Sophus" \
    -DBUILD_SHARED_LIBS=OFF \
    -DEIGEN3_INCLUDE_DIR="$LIB_ROOT/eigen"
  emmake make -j install
  popd >/dev/null
}

build_CERES(){
  rm -rf "$INSTALL_DIR/ceres-solver" "$LIB_ROOT/ceres-solver/build"
  mkdir -p "$LIB_ROOT/ceres-solver/build"
  pushd "$LIB_ROOT/ceres-solver/build" >/dev/null
  emcmake cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_TOOLCHAIN_FILE="$EMSCRIPTEN_CMAKE_DIR" \
    -DCMAKE_C_FLAGS="$C_FLAGS" \
    -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/ceres-solver" \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTING=OFF \
    -DEIGENSPARSE=ON \
    -DCERES_THREADING_MODEL=NO_THREADS \
    -DMINIGLOG=ON \
    -DEigen3_DIR="$INSTALL_DIR/eigen/share/eigen3/cmake"
  emmake make -j install

  # Patch includes (BSD sed on macOS)
  if [[ "$(uname)" == "Darwin" ]]; then
    find "$INSTALL_DIR/ceres-solver/include" -type f -name '*.h' -exec sed -i"" \
      's#glog/logging.h#ceres/internal/miniglog/glog/logging.h#g' {} +
  else
    find "$INSTALL_DIR/ceres-solver/include" -type f -name '*.h' -exec sed -i \
      's#glog/logging.h#ceres/internal/miniglog/glog/logging.h#g' {} +
  fi
  popd >/dev/null
}

build_OPENGV(){
  rm -rf "$INSTALL_DIR/opengv" "$LIB_ROOT/opengv/build"
  mkdir -p "$LIB_ROOT/opengv/build"
  pushd "$LIB_ROOT/opengv/build" >/dev/null
  emcmake cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=23 \
    -DCMAKE_TOOLCHAIN_FILE="$EMSCRIPTEN_CMAKE_DIR" \
    -DCMAKE_C_FLAGS="$C_FLAGS" \
    -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/opengv" \
    -DBUILD_SHARED_LIBS=OFF \
    -DEIGEN_INCLUDE_DIR="$LIB_ROOT/eigen"
  emmake make -j install
  popd >/dev/null
}

build(){
  local targets=("$@")
  local total=${#targets[@]}
  local BLUE='\033[1;34m' NC='\033[0m'

  for i in "${!targets[@]}"; do
    echo -e "${BLUE}[${i}/${total}] ------------ Building: ${targets[i]} ${NC}"
    "build_${targets[i]}"
    echo -e "${BLUE}[${i}/${total}] ------------ ✔ Done ${NC}\n"
  done
}

libs_to_build=(EIGEN OPENCV OBINDEX2 IBOW_LCD SOPHUS CERES OPENGV)
build "${libs_to_build[@]}"