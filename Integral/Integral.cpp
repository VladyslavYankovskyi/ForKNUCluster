#include <mpi.h>
#include <stdio.h>
#include <math.h>

// Функція для інтегрування
double f(double x) {
    return sin(x) * exp(-x * x) + cos(x * x) * log(x + 1);
}

// Функція для розрахунку інтегралу методом трапецій
double trapezoidal_method(double local_a, double local_b, int local_n, double h) {
    double integral;
    double x;
    int i;

    integral = (f(local_a) + f(local_b)) / 2.0;
    x = local_a;
    for (i = 1; i <= local_n-1; i++) {
        x = x + h;
        integral = integral + f(x);
    }
    integral = integral * h;
    return integral;
}

int main(int argc, char** argv) {
    
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 1000000;
    double a = 0.0, b = 1.0;
    double local_int, total_int;
    double start_time, end_time, parallel_time;

    double h = (b - a) / n;  // крок
    int local_n = n / size;  // кількість інтервалів для кожного процесу

    double local_a = a + rank * local_n * h;
    double local_b = local_a + local_n * h;

    // Вимірювання часу для паралельного виконання
    start_time = MPI_Wtime();
    
    local_int = trapezoidal_method(local_a, local_b, local_n, h);

    MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    end_time = MPI_Wtime();
    parallel_time = end_time - start_time;

    if (rank == 0) {
        printf("Значення інтегралу від %f до %f дорівнює %f\n", a, b, total_int);
        printf("Час виконання паралельної програми: %f секунд\n", parallel_time);
    }

    MPI_Finalize();
    return 0;
}