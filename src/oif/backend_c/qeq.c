#include <math.h>

int solve_qeq(double a, double b, double c, double roots[2])
{
    if (a == 0.0) {
        roots[0] = -c / b;
        roots[1] = -c / b;
    } else {
        double D = pow(b, 2.0) - 4 * a* c;
        if (b > 0) {
            roots[0] = -b - sqrt(D);
            roots[1] = c / (a * roots[0]);
        } else {
            roots[0] = -b + sqrt(D);
            roots[1] = c / (a * roots[0]);
        }
    }

    return 0;
}
