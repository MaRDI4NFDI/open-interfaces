#include <stdint.h>
#include <stdio.h>

#include <oif/api.h>
#include <oif/frontend_c/qeq.h>


int main(int argc, char *argv[])
{
    printf("Calling open interfaces from C\n");

    double a = 1.0;
    double b = 5.0;
    double c = 4.0;

    intptr_t dimensions[] = {2,};

    BackendHandle bh = oif_init_backend("c", "", 1, 0);

    printf("Solving quadratic equation a x^2 + b x + c = 0\n");
    printf("for a = %g, b = %g, c = %g\n", a, b, c);
    OIFArray *roots = create_array_f64(1, dimensions);
    int status = oif_solve_qeq(bh, a, b, c, roots);
    if (status) {
        fprintf(stderr, "Error occurred\n");
    }
    printf("x1 = %g\n", ((double *) roots->data)[0]);
    printf("x2 = %g\n", ((double *) roots->data)[1]);

    free_array(roots);

    return 0;
}
