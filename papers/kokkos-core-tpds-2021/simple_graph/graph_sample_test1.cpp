#define CU_INIT_UUID_STATIC

#include "helpers/helper_cuda.h"
#include <cuda.h>
#include "helpers/cuda_etbl/cuda_graphs.h"

extern "C" __host__ __device__ void __cuda_syscall_cuGraphSetConditional(unsigned long long handle, bool value);

#include <Kokkos_Core.hpp>
#include <Kokkos_Graph.hpp>

#include <vector>
#include <cstdio>
#include <cstdlib>

using Vector_t = Kokkos::View<double*>;
using Scalar_t = Kokkos::View<double>;

////////////////////////////////////////////////////
// Sample program A
////////////////////////////////////////////////////

template <class ExecSpace, typename ViewType>
struct Axpby {
  Vector_t x, y;
  double alpha;
  ViewType beta;
  KOKKOS_FUNCTION
  void operator()(const int& i) const noexcept {
    x(i) = alpha * x(i) + beta() * y(i);
  }
};

template <class ExecSpace, class T>
struct Dot {
  Vector_t x, y;
  
  KOKKOS_FUNCTION
  void operator()(const int& i, T& lsum) const noexcept { lsum += x(i) * y(i); }
  KOKKOS_FUNCTION
  void final(double& result) const {
    while (result > 1.0) {
      result /= 800000.0;
    }
  }
};

void warmup(uint32_t N) {
  Vector_t x("x", N), y("y", N), z("z", N);
  Scalar_t dotp("dotp");
  double alpha = 1.0;
  Scalar_t beta("beta");
  using EXECSPACE     = Kokkos::DefaultExecutionSpace;
  using axpby_functor = Axpby<EXECSPACE, Scalar_t>;
  EXECSPACE ex{};

  Kokkos::deep_copy(ex, dotp, 0);
  Kokkos::deep_copy(ex, x, 1);
  Kokkos::deep_copy(ex, y, 2);
  Kokkos::deep_copy(ex, z, 3);
  ex.fence();

  auto graph = Kokkos::Experimental::create_graph(ex, [&](auto root) {
    for (int i = 0; i < 1000; i++) {
      root.then_parallel_for(N, axpby_functor{x, y, alpha, beta});
    }
  });

  for (uint32_t i = 0; i < 10; i++) {
    graph.submit();
    ex.fence();
  }
}

static __managed__ struct {
  cuuint64_t conditionalHandle;
} loopData;


template <class ExecSpace, typename ViewType>
struct ConditionalNode {
  ViewType beta;
  Kokkos::View<int> counter;
  uint32_t iters;
  KOKKOS_FUNCTION
  void operator()(const int& i) const noexcept {
    //printf("beta=%.10f, counter=%d\n", beta(), counter());
    if (beta() != 0 /*0.000000000001*/ and ++counter() < iters) {
      __cuda_syscall_cuGraphSetConditional(loopData.conditionalHandle, true);
    }
  }
};

static __global__ void kernel()
{
  //printf("Hello from loop kernel!\n");
  __cuda_syscall_cuGraphSetConditional(loopData.conditionalHandle, true);
}

template <typename Ex, typename Beta>
auto runGraph(Ex ex, Beta beta, Vector_t x, Vector_t y, Vector_t z, uint32_t const iters, uint32_t const N) {
  using EXECSPACE     = Kokkos::DefaultExecutionSpace;
  using axpby_functor = Axpby<EXECSPACE, Beta>;
  using dotp_functor  = Dot<EXECSPACE, double>;
  using conditional_functor = ConditionalNode<EXECSPACE, Beta>;

  double alpha = 1.0, gamma = 0.6;
  Kokkos::View<int> counter("counter");
  Kokkos::deep_copy(ex, counter, 0);
  ex.fence();
  Kokkos::Timer full_timer;

  CUetblCudaGraphs *graphsTable;
  checkCudaErrors((cudaError_t)cuGetExportTable((const void **)&graphsTable, &CU_ETID_CudaGraphs));
  cudaGraphNode_t nodeInnerEntry, nodeInnerConditional;
  cudaGraph_t graph;
  struct cudaKernelNodeParams kernelNodeParams = { NULL };
  
  // Construct graph
  auto iteration = Kokkos::Experimental::create_graph(ex, [&](auto root) {
    auto f_xpy = root.then_parallel_for(N, axpby_functor{x, y, alpha, beta});
    auto f_zpy = root.then_parallel_for(N, axpby_functor{z, y, gamma, beta});
    auto ready = when_all(f_xpy, f_zpy);
    auto after_ready = ready.then_parallel_reduce(N, dotp_functor{x, z}, beta);
    after_ready.then_parallel_for(1, conditional_functor{beta, counter, iters});
  });

  auto impl = iteration.getImpl();
  auto iteration_graph = impl->m_graph;

  checkCudaErrors(cudaGraphCreate(&graph, 0));
  kernelNodeParams.gridDim.x = kernelNodeParams.gridDim.y = kernelNodeParams.gridDim.z = 1;
  kernelNodeParams.blockDim.x = kernelNodeParams.blockDim.y = kernelNodeParams.blockDim.z = 1;
  kernelNodeParams.func = (void *)kernel;
  checkCudaErrors(cudaGraphAddKernelNode(&nodeInnerEntry, graph, NULL, 0, &kernelNodeParams));
  checkCudaErrors((cudaError_t)graphsTable->cuGraphAddConditionalNode(&nodeInnerConditional, graph, &nodeInnerEntry, 1, iteration_graph));

  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  checkCudaErrors((cudaError_t)graphsTable->cuGraphExecGetConditionalNodeSchedulingHandle(graphExec, nodeInnerConditional, &loopData.conditionalHandle));

  Kokkos::Timer part_timer;

  checkCudaErrors(cudaGraphLaunch(graphExec, 0));
  ex.fence();
  auto x_ = std::make_tuple(full_timer.seconds(), part_timer.seconds(), 0);
  printf("Graph execution finished.\n");
  return x_;
}

int main(int argc, char* argv[]) {
  uint32_t const use_graph = atoi(argv[1]);
  uint32_t const N = atoi(argv[2]);
  uint32_t const iters = atoi(argv[3]);
  uint32_t const type = atoi(argv[4]);

  printf("use_graph=%u, N=%u, iters=%u\n", use_graph, N, iters);
  fflush(stdout);

  {
    Kokkos::ScopeGuard guard(argc, argv);

    warmup(16384);
    
    Vector_t x("x", N), y("y", N), z("z", N);
    Scalar_t dotp("dotp");
    Scalar_t beta_device("beta");
    Kokkos::View<double,Kokkos::HostSpace> beta_host("beta host");
    using EXECSPACE     = Kokkos::DefaultExecutionSpace;
    using dotp_functor  = Dot<EXECSPACE, double>;
    EXECSPACE ex{};

    // Set up
    Kokkos::deep_copy(ex, dotp, 0);
    Kokkos::deep_copy(ex, x, 1);
    Kokkos::deep_copy(ex, y, 2);
    Kokkos::deep_copy(ex, z, 3);
    Kokkos::deep_copy(ex, beta_device, 0.1);
    ex.fence();

    printf("Setup complete\n");
    fflush(stdout);

    double end_time = 0.;
    double part_end_time = 0.;

    std::tuple<double, double, double> out;
    out = runGraph(ex, beta_device, x, y, z, iters, N);
    end_time = std::get<0>(out);
    part_end_time = std::get<1>(out);
    printf("out=%f\n", std::get<2>(out));

    printf("full_time=%lf, part_time=%lf\n", end_time, part_end_time);
  }

  return 0;
}
