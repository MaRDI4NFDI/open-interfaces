#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <oif/frontend_c/frontend_c.h>
#include <oif/frontend_c/qeq.h>


char *parse_backend(int argc, char *argv[]) {
    if (argc == 1) {
        return "c";
    } else {
        if ((strcmp(argv[1], "c") == 0) || (strcmp(argv[1], "python") == 0)) {
            return argv[1];
        } else {
            fprintf(stderr, "USAGE: %s [c|python]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
}


int main(int argc, char *argv[])
{
    char *backend = parse_backend(argc, argv);
    printf("Calling from C an open interface for quadratic solver \n");
    printf("Backend: %s\n", backend);

    double a = 1.0;
    double b = 5.0;
    double c = 4.0;

    BackendHandle bh = oif_init_backend(backend, "", 1, 0);
    if (bh == OIF_BACKEND_INIT_ERROR) {
        fprintf(stderr, "Error during backend initialization. Cannot proceed\n");
        return EXIT_FAILURE;
    }

    printf("Solving quadratic equation a x^2 + b x + c = 0\n");
    printf("for a = %g, b = %g, c = %g\n", a, b, c);

    intptr_t dimensions[] = {2,};
    OIFArrayF64 *roots = create_array_f64(1, dimensions);
    int status = oif_solve_qeq(bh, a, b, c, roots);
    if (status) {
        free_array_f64(roots);
        return EXIT_FAILURE;
    }

    printf("x1 = %g\n", ((double *) roots->data)[0]);
    printf("x2 = %g\n", ((double *) roots->data)[1]);

    free_array_f64(roots);

    return 0;
}
