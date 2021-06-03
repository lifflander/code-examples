
#include <Kokkos_Core.hpp>
#include <Kokkos_Graph.hpp>

#include <vector>
#include <cstdio>
#include <cstdlib>

#include <mpi.h>

using Vector_t = Kokkos::View<double*>;
using Scalar_t = Kokkos::View<double>;

template <class ExecSpace>
struct Axpby {
  Vector_t x, y;
  double alpha, beta;
  KOKKOS_FUNCTION
  void operator()(const int& i) const noexcept {
    x(i) = alpha * x(i) + beta * y(i);
  }
};

template <class ExecSpace, class T>
struct Dot {
  Vector_t x, y;
  KOKKOS_FUNCTION
  void operator()(const int& i, T& lsum) const noexcept { lsum += x(i) * y(i); }
};

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  
  uint32_t const use_graph = atoi(argv[1]);
  uint32_t const N = atoi(argv[2]);
  uint32_t const iters = atoi(argv[3]);

  printf("use_graph=%u, N=%u, iters=%u\n", use_graph, N, iters);
  fflush(stdout);

  {
    Kokkos::ScopeGuard guard(argc, argv);

    Vector_t x("x", N), y("y", N), z("z", N);
    Scalar_t dotp("dotp");
    double alpha = 1.0, beta = 0.8, gamma = 0.6;
    using EXECSPACE     = Kokkos::DefaultExecutionSpace;
    using axpby_functor = Axpby<EXECSPACE>;
    using dotp_functor  = Dot<EXECSPACE, double>;
    EXECSPACE ex{};

    // Set up
    Kokkos::deep_copy(ex, dotp, 0);
    Kokkos::deep_copy(ex, x, 1);
    Kokkos::deep_copy(ex, y, 2);
    Kokkos::deep_copy(ex, z, 3);
    ex.fence();

    printf("Setup complete\n");
    fflush(stdout);

    double end_time = 0.;
    double part_end_time = 0.;

    if (use_graph == 1) {
      Kokkos::Timer full_timer;
      Kokkos::Timer part_timer;

      // Construct graph
      auto graph = Kokkos::Experimental::create_graph(ex, [&](auto root) {
        auto f_xpy = root.then_parallel_for(N, axpby_functor{x, y, alpha, beta});
        auto f_zpy = root.then_parallel_for(N, axpby_functor{z, y, gamma, beta});
        auto ready = when_all(f_xpy, f_zpy);
        ready.then_parallel_reduce(N, dotp_functor{x, z}, dotp);
      });

      // Submit many times
      for (uint32_t i = 0; i < iters; i++) {
      	if (i % 10 == 0 or i == 0) {
      	  printf("Starting iteration %d...\n", i);
      	  fflush(stdout);
      	}

      	if (i == 1) {
      	  part_timer.reset();
      	}
    
      	graph.submit();
      	ex.fence();
      }

      end_time = full_timer.seconds();
      part_end_time = part_timer.seconds();
    } else if (use_graph == 0) {
      Kokkos::Timer full_timer;
      Kokkos::Timer part_timer;

      for (uint32_t i = 0; i < iters; i++) {
      	if (i % 10 == 0 or i == 0) {
      	  printf("Starting iteration %d...\n", i);
      	  fflush(stdout);
      	}

      	if (i == 1) {
      	  part_timer.reset();
      	}

      	Kokkos::parallel_for(N, axpby_functor{x, y, alpha, beta});
      	Kokkos::parallel_for(N, axpby_functor{z, y, gamma, beta});
      	Kokkos::parallel_reduce(N, dotp_functor{x, z}, dotp);
      	ex.fence();
      }

      end_time = full_timer.seconds();
      part_end_time = part_timer.seconds();
    } else if (use_graph == 2) {
      using DeviceSpace = Kokkos::DefaultExecutionSpace;

      std::vector<DeviceSpace> space;
      for (int i = 0; i < 2; i++) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        space.push_back(DeviceSpace(stream));
      }
      
      auto N_policy_1 =
        Kokkos::Experimental::require(
          Kokkos::RangePolicy<>(space[0], 0, N),
      	  Kokkos::Experimental::WorkItemProperty::HintLightWeight
        );
      auto N_policy_2 =
        Kokkos::Experimental::require(
          Kokkos::RangePolicy<>(space[1], 0, N),
      	  Kokkos::Experimental::WorkItemProperty::HintLightWeight
        );

      Kokkos::Timer full_timer;
      Kokkos::Timer part_timer;

      for (uint32_t i = 0; i < iters; i++) {
        if (i % 10 == 0 or i == 0) {
	  printf("Starting iteration %d...\n", i);
	  fflush(stdout);
        }

        if (i == 1) {
	  part_timer.reset();
        }

        Kokkos::parallel_for(N_policy_1, axpby_functor{x, y, alpha, beta});
        Kokkos::parallel_for(N_policy_2, axpby_functor{z, y, gamma, beta});
	cudaEvent_t event;
	cudaEventCreate(&event);
	cudaEventRecord(event, space[1].cuda_stream());
	cudaStreamWaitEvent(space[0].cuda_stream(), event, 0);
	// space[1].fence();
	// space[1].fence();
        Kokkos::parallel_reduce(N_policy_1, dotp_functor{x, z}, dotp);
	space[0].fence();
	// space[1].fence();
      }

      end_time = full_timer.seconds();
      part_end_time = part_timer.seconds();
    } else {
      fprintf(stderr, "FAILURE\n");
    }

    printf("full_time=%lf, part_time=%lf\n", end_time, part_end_time);
  }
  
  MPI_Finalize();
  return 0;
}
