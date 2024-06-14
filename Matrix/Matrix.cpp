#include <mpi.h>
#include <iostream>
#include <vector>
#include <random> // for std::mt19937 and std::uniform_int_distribution
#include <iomanip> // for std::setw, std::setprecision

using namespace std;

void multiply_matrices(const vector<int>& A, const vector<int>& B, vector<int>& C, int N, int start_row, int end_row) {
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < N; ++j) {
            C[(i - start_row) * N + j] = 0;
            for (int k = 0; k < N; ++k) {
                C[(i - start_row) * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

void print_matrix(const vector<int>& matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << setw(5) << matrix[i * N + j] << " ";
        }
        cout << endl;
    }
}

void randomize_matrix(vector<int>& matrix, int N) {
    random_device rd;  // obtain a random number from hardware
    mt19937 eng(rd()); // seed the generator

    uniform_int_distribution<> distr(1, 10); // define the range

    for (int i = 0; i < N * N; ++i) {
        matrix[i] = distr(eng); // generate random numbers and assign to matrix elements
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time, end_time;
    if (rank == 0) {
        start_time = MPI_Wtime(); // Start timing the program
    }

    int N = 300; // Size of square matrices A and B
    vector<int> A, B, C;

    if (rank == 0) {
        A.resize(N * N);
        B.resize(N * N);
        C.resize(N * N);

        // Randomize matrices A and B with values from 1 to 10
        randomize_matrix(A, N);
        randomize_matrix(B, N);
    }

    // Broadcast matrix dimension N
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast matrices A and B to all processes
    if (rank != 0) {
        A.resize(N * N);
        B.resize(N * N);
    }
    MPI_Bcast(A.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Divide rows of C among processes
    int rows_per_process = N / size;
    int extra_rows = N % size;
    int start_row = rank * rows_per_process + min(rank, extra_rows);
    int end_row = start_row + rows_per_process + (rank < extra_rows ? 1 : 0);

    // Allocate local result matrix for each process
    vector<int> local_C((end_row - start_row) * N);

    // Each process computes its part of C
    multiply_matrices(A, B, local_C, N, start_row, end_row);

    // Prepare to gather the results
    vector<int> sendcounts(size);
    vector<int> displs(size);
    int offset = 0;

    for (int i = 0; i < size; ++i) {
        sendcounts[i] = ((i < extra_rows) ? (rows_per_process + 1) : rows_per_process) * N;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    if (rank == 0) {
        C.resize(N * N);
    }

    // Gather the results from all processes
    MPI_Gatherv(local_C.data(), sendcounts[rank], MPI_INT, C.data(), sendcounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Print the result matrix C to the console
        cout << "Result matrix C:" << endl;
        // print_matrix(C, N);

        // End timing and calculate elapsed time
        end_time = MPI_Wtime();
        double elapsed_time = end_time - start_time;
        cout << "Elapsed time: " << setprecision(6) << fixed << elapsed_time << " seconds." << endl;
    }

    MPI_Finalize();
    return 0;
}