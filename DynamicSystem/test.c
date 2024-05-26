
#include <stdio.h>
#include <math.h>

// Function to print a matrix
void print_matrix(const char* desc, double* matrix, int rows, int cols) {
    printf("\n%s\n", desc);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

// Function to calculate SVD for a 2x2 matrix
void svd_2x2(double* a, double* U, double* S, double* V) {
    double A = a[0];
    double B = a[1];
    double C = a[2];
    double D = a[3];

    double E = (A + D) / 2.0;
    double F = (A - D) / 2.0;
    double G = (C + B) / 2.0;
    double H = (C - B) / 2.0;

    double Q = sqrt(E * E + H * H);
    double R = sqrt(F * F + G * G);

    double theta = atan2(G, F);
    double phi = atan2(H, E);

    S[0] = Q + R;
    S[1] = Q - R;

    U[0] = cos(theta);
    U[1] = sin(theta);
    U[2] = -sin(theta);
    U[3] = cos(theta);

    V[0] = cos(phi);
    V[1] = -sin(phi);
    V[2] = sin(phi);
    V[3] = cos(phi);
}

int main() {
    // Define a 2x2 matrix
    double a[4] = {
        1.0, 2.0,
        3.0, 4.0
    };

    // Allocate memory for the singular values and the left and right singular vectors
    double U[4], S[2], V[4];

    // Compute SVD
    svd_2x2(a, U, S, V);

    // Print the results
    print_matrix("Singular values (S):", S, 1, 2);
    print_matrix("Left singular vectors (U):", U, 2, 2);
    print_matrix("Right singular vectors (V^T):", V, 2, 2);

    return 0;
}
