#include <stdio.h>

#include <oif/api.h>


int main(int argc, char *argv[])
{
    printf("Calling open interfaces from C\n");

    double a = 1.0;
    double b = 5.0;
    double c = 4.0;

    OIFArray *x = create_array_f64(1, {2,});

    BackendHandle bh = init_backend("c", 1, 0);

    printf("Solving quadratic equation for a = %g, b = %g, c = %g\n", a, b, c);
    int status = solve_qeq(bh, a, b, c, roots);
    if (status) {
        fprintf(stderr, "Error occurred\n");
    }
    printf("x1 = %g", x[0]);
    printf("x2 = %g", x[1]);

    free_array(x);

    return 0;
}
