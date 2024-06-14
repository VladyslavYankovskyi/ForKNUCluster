#include <iostream>
#include <cmath>
#include <mpi.h>

using namespace std;

// Function to integrate (example function: f(x, y) = x^2 + y^2)
double f(double x, double y) {
    return x * x * x + y * y;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time, end_time;

    // Start measuring time
    start_time = MPI_Wtime();

    double a = 0.0;  // Lower bound for x
    double b = 1.0;  // Upper bound for x
    double c = 0.0;  // Lower bound for y
    double d = 1.0;  // Upper bound for y
    int N = 10000;   // Number of intervals for both x and y

    double dx = (b - a) / N;
    double dy = (d - c) / N;

    // Calculate local sum
    double local_sum = 0.0;
    for (int i = rank; i < N; i += size) {
        double x = a + (i + 0.5) * dx;
        for (int j = 0; j < N; j++) {
            double y = c + (j + 0.5) * dy;
            local_sum += f(x, y);
        }
    }
    local_sum *= dx * dy;

    // Reduce all local sums to get the global sum
    double global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // End measuring time
    end_time = MPI_Wtime();

    // Print the result and execution time in rank 0
    if (rank == 0) {
        cout << "Double integral value: " << global_sum << endl;
        cout << "Execution time: " << end_time - start_time << " seconds" << endl;
    }

    MPI_Finalize();

    return 0;
}