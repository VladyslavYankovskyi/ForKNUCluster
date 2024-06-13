#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

// Function to estimate Pi using the Monte Carlo method
double estimate_pi(long num_samples) {
    long count = 0;
    for (long i = 0; i < num_samples; ++i) {
        double x = static_cast<double>(rand()) / RAND_MAX;
        double y = static_cast<double>(rand()) / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            ++count;
        }
    }
    return 4.0 * count / num_samples;
}

int main(int argc, char* argv[]) {
    int rank, size;
    long num_samples = 10000000; // Default number of samples per process

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 1) {
        num_samples = atol(argv[1]) / size;
    }

    srand(time(NULL) + rank); // Seed the random number generator differently for each process

    // Start the timer
    double start_time = MPI_Wtime();

    // Each process estimates Pi using its portion of samples
    double local_pi = estimate_pi(num_samples);

    if (rank == 0) {
        double global_pi = local_pi;
        double temp;

        // Receive results from all other processes and sum them up
        for (int i = 1; i < size; ++i) {
            MPI_Recv(&temp, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_pi += temp;
        }

        // Calculate the average value of Pi
        global_pi /= size;

        // Stop the timer
        double end_time = MPI_Wtime();

        // Print the result
        std::cout << "Estimated value of Pi: " << global_pi << std::endl;
        std::cout << "Execution time: " << end_time - start_time << " seconds" << std::endl;
    } else {
        // Send the local estimate of Pi to the root process
        MPI_Send(&local_pi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}