#include <math.h>
#include <stdio.h>
#include <stdlib.h>


/**
 * Solve quadratic equation ax**2 + bx + c = 0
 * Assumes that the output array `roots` always has two elements,
 * so if the roots are repeated, they both will be still present.
 * @param a Coefficient before x**2
 * @param b Coefficient before x
 * @param c Free term
 * @param roots Array with two elements to which the found roots are written.
 * @return int
 */
int solve_qeq(double a, double b, double c, double *roots)
{
    if (roots == NULL) {
        fprintf(stderr, "Memory for the roots array was not provided\n");
        exit(EXIT_FAILURE);
    }
    if (a == 0.0) {
        roots[0] = -c / b;
        roots[1] = -c / b;
    } else {
        double D = pow(b, 2.0) - 4 * a* c;
        if (b > 0) {
            roots[0] = (-b - sqrt(D)) / (2 * a);
            roots[1] = c / (a * roots[0]);
        } else {
            roots[0] = (-b + sqrt(D)) / (2 * a);
            roots[1] = c / (a * roots[0]);
        }
    }

    return 0;
}
