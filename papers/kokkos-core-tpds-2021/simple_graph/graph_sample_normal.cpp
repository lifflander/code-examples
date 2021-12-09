
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

template <class ExecSpace>
struct Axpby {
  Vector_t x, y;
  double alpha;
  double beta;
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

void warmup(uint32_t N) {
  Vector_t x("x", N), y("y", N), z("z", N);
  Scalar_t dotp("dotp");
  double alpha = 1.0, beta = 0.8;
  using EXECSPACE     = Kokkos::DefaultExecutionSpace;
  using axpby_functor = Axpby<EXECSPACE>;
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

int main(int argc, char* argv[]) {
  uint32_t const use_graph = atoi(argv[1]);
  uint32_t const N = atoi(argv[2]);
  uint32_t const iters = atoi(argv[3]);

  printf("use_graph=%u, N=%u, iters=%u\n", use_graph, N, iters);
  fflush(stdout);

  {
    Kokkos::ScopeGuard guard(argc, argv);

    warmup(16384);

    Vector_t x("x", N), y("y", N), z("z", N);
    Scalar_t dotp("dotp");
    double alpha = 1.0, gamma = 0.6, beta = 0.8;
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
      auto iteration = Kokkos::Experimental::create_graph(ex, [&](auto root) {
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

	iteration.submit();
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
    } else {
      fprintf(stderr, "FAILURE\n");
    }

    printf("full_time=%lf, part_time=%lf\n", end_time, part_end_time);
  }

  return 0;
}
