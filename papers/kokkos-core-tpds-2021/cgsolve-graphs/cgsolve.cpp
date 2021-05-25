// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
// @HEADER

/*
  Adapted from the Mantevo Miniapp Suite.
  https://mantevo.github.io/pdfs/MantevoOverview.pdf
*/

#include "generate_matrix.hpp"
#include "Kokkos_Graph.hpp"
#include <mpi.h>

using vector_t = Kokkos::View<double*>;
using scalar_t = Kokkos::View<double>;

bool use_graph = false;

struct SPMV {
  vector_t y;
  CrsMatrix<typename Kokkos::DefaultExecutionSpace::memory_space> A;
  vector_t x;
  int rows_per_team;
  int64_t nrows;

  SPMV( vector_t& y_, CrsMatrix<typename Kokkos::DefaultExecutionSpace::memory_space>& A_,
        vector_t& x_,
        int rows_per_team_): y(y_),A(A_),x(x_),rows_per_team(rows_per_team_),nrows(y.extent(0)) {}

  KOKKOS_FUNCTION
  void operator() (const Kokkos::TeamPolicy<>::member_type& team) const {
        const int64_t first_row = team.league_rank() * rows_per_team;
        const int64_t last_row  = first_row + rows_per_team < nrows
                                      ? first_row + rows_per_team
                                      : nrows;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, first_row, last_row),
                             [&](const int64_t row) {
                               const int64_t row_start = A.row_ptr(row);
                               const int64_t row_length =
                                   A.row_ptr(row + 1) - row_start;

                               double y_row;
                               Kokkos::parallel_reduce(
                                   Kokkos::ThreadVectorRange(team, row_length),
                                   [=](const int64_t i, double& sum) {
                                     sum += A.values(i + row_start) *
                                            x(A.col_idx(i + row_start));
                                   },
                                   y_row);
                               y(row) = y_row;
                             });
      }
};

template <class YType, class AType, class XType>
void spmv(YType y, AType A, XType x) {

  // For low thread counts spread rows over individual threads
  int rows_per_team = 512;
  int team_size     = 1;

  // For high concurrency architecture use teams
  if(Kokkos::DefaultExecutionSpace().concurrency() > 1024) {
    rows_per_team = 16;
    team_size     = 16;
  }

  int64_t nrows = y.extent(0);
  Kokkos::parallel_for(
    "SPMV",
    Kokkos::TeamPolicy<>((nrows + rows_per_team - 1) / rows_per_team,
                           team_size, 8),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        const int64_t first_row = team.league_rank() * rows_per_team;
        const int64_t last_row  = first_row + rows_per_team < nrows
                                      ? first_row + rows_per_team
                                      : nrows;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, first_row, last_row),
                             [&](const int64_t row) {
                               const int64_t row_start = A.row_ptr(row);
                               const int64_t row_length =
                                   A.row_ptr(row + 1) - row_start;

                               double y_row;
                               Kokkos::parallel_reduce(
                                   Kokkos::ThreadVectorRange(team, row_length),
                                   [=](const int64_t i, double& sum) {
                                     sum += A.values(i + row_start) *
                                            x(A.col_idx(i + row_start));
                                   },
                                   y_row);
                               y(row) = y_row;
                             });
      });
}

struct Dot {
  vector_t x,y;
  scalar_t alpha,oldrtrans;
  Kokkos::View<double> rtrans;
  int invert;

  Dot(vector_t& x_, vector_t& y_, scalar_t& alpha_, Kokkos::View<double>& rtrans_, scalar_t& oldrtrans_, int invert_):
    x(x_),y(y_),alpha(alpha_),rtrans(rtrans_),oldrtrans(oldrtrans_), invert(invert_) {}

  KOKKOS_FUNCTION
  void operator() (const int64_t& i, double& lsum) const { lsum += y(i) * x(i); }
  KOKKOS_FUNCTION
  void final(double& result) const {
    if(invert) {
      alpha() = rtrans() / result;
      oldrtrans() = result;
    } else {
      oldrtrans() = rtrans();
      alpha() = result / oldrtrans();
      rtrans() = result;
    }
  }
};
template <class YType, class XType>
double dot(YType y, XType x) {
  double result;
  Kokkos::parallel_reduce(
      "DOT", y.extent(0),
      KOKKOS_LAMBDA(const int64_t& i, double& lsum) { lsum += y(i) * x(i); },
      result);
  return result;
}

struct AXPBY {
  vector_t z,x,y;
  scalar_t beta;
  double alpha,scale;
  AXPBY(vector_t& z_,  double a, vector_t& x_, double s, scalar_t& b, vector_t& y_):
          z(z_),alpha(a),x(x_),scale(s),beta(b),y(y_) {}

  KOKKOS_FUNCTION
  void operator() (const int& i) const { z(i) = alpha * x(i) + scale * beta() * y(i); }
};

template <class ZType, class YType, class XType>
void axpby(ZType z, double alpha, XType x, double scale, scalar_t beta, YType y) {
  int64_t n = z.extent(0);
  Kokkos::parallel_for(
      "AXPBY", n,
      KOKKOS_LAMBDA(const int& i) { z(i) = alpha * x(i) + scale * beta() * y(i); });
}

template <class VType, class AType>
int cg_solve(VType y, AType A, VType b, int max_iter, double tolerance, int64_t print_freq) {
  int myproc    = 0;
  int num_iters = 0;

  double normr     = 0;
  Kokkos::View<double> rtrans("RTrants");
  Kokkos::View<double,Kokkos::HostSpace> rtrans_host("RTrants");
  Kokkos::View<double,Kokkos::HostSpace> alpha_host("AlphaHost");
  scalar_t alpha("Alpha");
  scalar_t beta("Alpha");
  scalar_t p_ap_dot("p_Ap_dot");
  scalar_t oldrtrans("OldTrans");

  Kokkos::View<double, Kokkos::HostSpace> p_ap_dot_host("p_Ap_dot_host");

  VType x("x", b.extent(0));
  VType r("r", x.extent(0));
  VType p("r", x.extent(0));
  VType Ap("r", x.extent(0));
  double one  = 1.0;
  double zero = 0.0;

  alpha_host() = 1.;
  Kokkos::deep_copy(alpha, alpha_host);

  axpby(p, one, x, zero, alpha, x);
  spmv(Ap, A, p);
  axpby(r, one, b, -one, alpha, Ap);

  rtrans_host() = dot(r, r);

  normr = std::sqrt(rtrans_host());

  if (myproc == 0) {
    std::cout << "Initial Residual = " << normr << std::endl;
  }

  double brkdown_tol = std::numeric_limits<double>::epsilon();

  // Do iteration k == 1
  {
    int k = 1;
    axpby(p, one, r, zero, beta, r);
    spmv(Ap, A, p);
    Kokkos::deep_copy(rtrans, rtrans_host);
    Kokkos::parallel_reduce("Dot2",Ap.extent(0),Dot(Ap,p,alpha,rtrans,p_ap_dot,1));

    if (myproc == 0 && (k % print_freq == 0 || k == max_iter)) {
      Kokkos::deep_copy(rtrans_host, rtrans);
      normr = std::sqrt(rtrans());
      std::cout << "Iteration = " << k << "   Residual = " << normr
                << std::endl;
    }

    Kokkos::deep_copy(p_ap_dot_host, p_ap_dot);

    if (p_ap_dot_host() < brkdown_tol) {
      if (p_ap_dot_host() < 0) {
        std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!"
                  << std::endl;
        return num_iters;
      } else
        brkdown_tol = 0.1 * p_ap_dot_host();
    }

    axpby(x, one, x, one, alpha, p);
    axpby(r, one, r, -one, alpha, Ap);
    num_iters = k;
  }

  int rows_per_team = 512;
  int team_size     = 1;

  std::cout << "Kokkos::DefaultExecutionSpace().concurrency() = " << Kokkos::DefaultExecutionSpace().concurrency() << std::endl;

  // For high concurrency architecture use teams
  if (Kokkos::DefaultExecutionSpace().concurrency() > 1024) {
    rows_per_team = 8;
    team_size     = 8;
  }
  int64_t nrows = y.extent(0);

  std::cout << "nrows = " << nrows << std::endl;
  std::cout << "r.extent(0) = " << r.extent(0) << std::endl;
  std::cout << "rows_per_team = " << rows_per_team << std::endl;
  std::cout << "team_size = " << team_size << std::endl;
  std::cout << "use_graph = " << use_graph << std::endl;

  auto per_iter = [&](int k){
    if (myproc == 0 && (k % print_freq == 0 || k == max_iter)) {
      Kokkos::fence();
      Kokkos::deep_copy(rtrans_host, rtrans);
      normr = std::sqrt(rtrans_host());
      std::cout << "Iteration = " << k << "   Residual = " << normr
                << std::endl;

      Kokkos::deep_copy(p_ap_dot_host, p_ap_dot);
      if (p_ap_dot_host() < brkdown_tol) {
        if (p_ap_dot_host() < 0) {
          std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!"
                    << std::endl;
          exit(129);
        } else
          brkdown_tol = 0.1 * p_ap_dot_host();
      }
    } else {
      // if (k % 100 == 0) {
      Kokkos::fence();
      Kokkos::deep_copy(rtrans_host, rtrans);
      normr = std::sqrt(rtrans_host());
      // }
    }
  };

  if (use_graph) {
    auto graph = Kokkos::Experimental::create_graph(Kokkos::DefaultExecutionSpace(), [&] (auto root) {
	auto dot1   = root.  then_parallel_reduce("Dot1",r.extent(0),Dot(r,r,alpha,rtrans,oldrtrans,0),rtrans);
	auto axpby1 = dot1.  then_parallel_for("AXPBY",r.extent(0),AXPBY(p, one, r, one, alpha, p));
	auto spmvn  = axpby1.then_parallel_for("SPMV",
					       Kokkos::TeamPolicy<>((nrows + rows_per_team - 1) / rows_per_team,
								    team_size, 8),SPMV(Ap,A,p,rows_per_team));
	auto dot2   = spmvn.then_parallel_reduce("Dot2",Ap.extent(0),Dot(Ap,p,alpha,rtrans,p_ap_dot,1),p_ap_dot);
	auto axpby2 = dot2.then_parallel_for("AXPBY",x.extent(0),AXPBY(x, one, x, one, alpha, p));
	// For some reason its a bit slower if one actually exposes the available concurrency
	// Simple linear graph executes a bit faster
#if 0
	dot2.then_parallel_for("AXPBY",r.extent(0),AXPBY(r, one, r, -one, alpha, Ap));
#else
	axpby2.then_parallel_for("AXPBY",r.extent(0),AXPBY(r, one, r, -one, alpha, Ap));
#endif
      });

    for (int64_t k = 2; k <= max_iter && normr > tolerance; ++k) {
      graph.submit();
      Kokkos::fence();
      per_iter(k);
      num_iters = k;
    }
  } else {
    for (int64_t k = 2; k <= max_iter && normr > tolerance; ++k) {
      Kokkos::parallel_reduce("Dot1",r.extent(0),Dot(r,r,alpha,rtrans,oldrtrans,0),rtrans);
      Kokkos::parallel_for("AXPBY",r.extent(0),AXPBY(p, one, r, one, alpha, p));
      Kokkos::parallel_for("SPMV",
			   Kokkos::TeamPolicy<>((nrows + rows_per_team - 1) / rows_per_team,
						team_size, 8),SPMV(Ap,A,p,rows_per_team));
      Kokkos::parallel_reduce("Dot2",Ap.extent(0),Dot(Ap,p,alpha,rtrans,p_ap_dot,1),p_ap_dot);
      Kokkos::parallel_for("AXPBY",x.extent(0),AXPBY(x, one, x, one, alpha, p));
      Kokkos::parallel_for("AXPBY",r.extent(0),AXPBY(r, one, r, -one, alpha, Ap));
      per_iter(k);
      num_iters = k;
    }
  }
  return num_iters;
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  Kokkos::ScopeGuard guard(argc, argv);

  use_graph        = argc > 1 ? atoi(argv[1]) : 0;
  int N            = argc > 2 ? atoi(argv[2]) : 100;
  int max_iter     = argc > 3 ? atoi(argv[3]) : 200;
  double tolerance = argc > 4 ? atoi(argv[4]) : 1e-7;
  int64_t print_freq = argc > 5 ? atoi(argv[5]) : max_iter / 4;
  if (print_freq < 1) print_freq = 1;

  CrsMatrix<Kokkos::HostSpace> h_A = Impl::generate_miniFE_matrix(N);
  Kokkos::View<double*, Kokkos::HostSpace> h_x =
      Impl::generate_miniFE_vector(N);

  Kokkos::View<int64_t*> row_ptr("row_ptr", h_A.row_ptr.extent(0));
  Kokkos::View<int64_t*> col_idx("col_idx", h_A.col_idx.extent(0));
  Kokkos::View<double*> values("values", h_A.values.extent(0));

  CrsMatrix<Kokkos::DefaultExecutionSpace::memory_space> A(
      row_ptr, col_idx, values, h_A.num_cols());
  Kokkos::View<double*> x("X", h_x.extent(0));
  Kokkos::View<double*> y("Y", h_x.extent(0));

  Kokkos::deep_copy(x, h_x);
  Kokkos::deep_copy(A.row_ptr, h_A.row_ptr);
  Kokkos::deep_copy(A.col_idx, h_A.col_idx);
  Kokkos::deep_copy(A.values, h_A.values);

  Kokkos::Timer timer;
  int num_iters = cg_solve(y, A, x, max_iter, tolerance, print_freq);
  double time   = timer.seconds();

  // Compute Bytes and Flops
  double spmv_bytes = A.num_rows() * sizeof(int64_t) +
                      A.nnz() * sizeof(int64_t) + A.nnz() * sizeof(double) +
                      A.nnz() * sizeof(double) + A.num_rows() * sizeof(double);

  double dot_bytes   = x.extent(0) * sizeof(double) * 2;
  double axpby_bytes = x.extent(0) * sizeof(double) * 3;

  double spmv_flops  = A.nnz() * 2;
  double dot_flops   = x.extent(0) * 2;
  double axpby_flops = x.extent(0) * 3;

  int spmv_calls  = 1 + num_iters;
  int dot_calls   = num_iters;
  int axpby_calls = 2 + num_iters * 3;

  printf("CGSolve for 3D (%i %i %i); %i iterations; %lf time\n", N, N, N,
         num_iters, time);
  printf(
      "Performance: %lf GFlop/s %lf GB/s (Calls SPMV: %i Dot: %i AXPBY: %i\n",
      1e-9 *
          (spmv_flops * spmv_calls + dot_flops * dot_calls +
           axpby_flops * axpby_calls) /
          time,
      (1.0 / 1024 / 1024 / 1024) *
          (spmv_bytes * spmv_calls + dot_bytes * dot_calls +
           axpby_bytes * axpby_calls) /
          time,
      spmv_calls, dot_calls, axpby_calls);

  MPI_Finalize();
}
