#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>

// Function to integrate
double f(double x, double y) {
    return x * x * x + y * y;
}

int main(int argc, char** argv) {

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Integration limits
    double x_min = 0.0, x_max = 1.0;
    double y_min = 0.0, y_max = 1.0;

    // Number of random points
    long long num_points = 30000000;

    // Seed for random number generator
    std::srand(std::time(0) + rank);

    // Calculate the area of the integration region
    double area = (x_max - x_min) * (y_max - y_min);

    // Start timing
    double start_time = MPI_Wtime();

    // Monte Carlo integration
    long long points_per_proc = num_points / size;
    double local_sum = 0.0;

    for (long long i = 0; i < points_per_proc; ++i) {
        double x = x_min + (x_max - x_min) * (double)std::rand() / RAND_MAX;
        double y = y_min + (y_max - y_min) * (double)std::rand() / RAND_MAX;
        local_sum += f(x, y);
    }

    // Average value of the function over the domain
    local_sum = local_sum * area / points_per_proc;

    // Reduce the results to the root process
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Stop timing
    double end_time = MPI_Wtime();

    if (rank == 0) {
        // Calculate the average value over all processes
        global_sum = global_sum / size;
        std::cout << "Estimated integral: " << global_sum << std::endl;
        std::cout << "Execution time: " << end_time - start_time << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}