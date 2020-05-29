/*
 * Eigen benchmark of matrix exponential
 *
 * Created Date: Friday, May 29th 2020, 12:51:51 am
 * Author: Henry Liu
 */

#include <iostream>
#include <complex>
#include <chrono>

#define EIGEN_USE_MKL_ALL // Enables BLAS, LAPACK, and Intel MKL/VML
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

using Eigen::Array4i;
using Eigen::ArrayXcf;
using Eigen::MatrixXcf;

using namespace std::chrono;

int main(int argc, char *argv[])
{
    int N = std::stoi(argv[1]);

    MatrixXcf a(N, N);
    for (int j = 1; j < N; j++) // Generate lowering operator
        a(j - 1, j) = sqrt(j);
    MatrixXcf a_dag = a.adjoint(); // Generate raising operator

    // Batch of 100 alphas with amplitude 3 from 0 to pi
    std::complex<float> limit(0, M_PI);
    ArrayXcf displacements = ArrayXcf::LinSpaced(100, 0, limit);
    displacements = 3 * displacements.exp();

    MatrixXcf results[100];
    auto start = high_resolution_clock::now();

    // There's no "vectorization" like TF but OMP provides some speedup
    #pragma omp parallel for
    for (int i = 0; i < 100; i++)
    {
        auto alpha = displacements[i];
        results[i] = alpha * a_dag - std::conj(alpha) * a;
        results[i] = results[i].exp();
    }

    auto stop = high_resolution_clock::now();
    duration<double> duration = stop - start;
    std::cout << "Time: " << duration.count() << std::endl;
}

/* Compilation:
g++-9 -O3 -fopenmp --std=c++11 -I/usr/local/include/eigen3 -m64 -I${MKLROOT}/include \
${MKLROOT}/lib/libmkl_intel_lp64.a ${MKLROOT}/lib/libmkl_sequential.a ${MKLROOT}/lib/libmkl_core.a \
-lpthread -lm -ldl eigen_benchmark.cc -o eigen_benchmark.out
*/
