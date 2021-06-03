#!/usr/bin/env bash

cmake -GNinja -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ROOT=/vscratch1/jliffla/kokkos-benchmark/kokkos-install/ -DKokkosKernels_ROOT=/vscratch1/jliffla/kokkos-benchmark/kokkos-kernels-install ../
