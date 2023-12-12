#include "oif/api.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <oif/c_bindings.h>
#include <oif/interfaces/qeq.h>

char *parse_impl(int argc, char *argv[]) {
    if (argc == 1) {
        return "c_qeq_solver";
    } else {
        if ((strcmp(argv[1], "c_qeq_solver") == 0) ||
            (strcmp(argv[1], "py_qeq_solver") == 0)) {
            return argv[1];
        } else {
            fprintf(
                stderr, "USAGE: %s [c_qeq_solver|py_qeq_solver]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char *argv[]) {
    char *impl = parse_impl(argc, argv);
    printf("Calling from C an open interface for quadratic solver \n");
    printf("Implementation: %s\n", impl);

    double a = 1.0;
    double b = 5.0;
    double c = 4.0;

    ImplHandle implh = oif_init_impl("qeq", impl, 1, 0);
    if (implh == OIF_IMPL_INIT_ERROR) {
        fprintf(stderr,
                "Error during implementation initialization. Cannot proceed\n");
        return EXIT_FAILURE;
    }

    printf("Solving quadratic equation a x^2 + b x + c = 0\n");
    printf("for a = %g, b = %g, c = %g\n", a, b, c);

    intptr_t dimensions[] = {
        2,
    };
    OIFArrayF64 *roots = oif_create_array_f64(1, dimensions);
    int status = oif_solve_qeq(implh, a, b, c, roots);
    if (status) {
        oif_free_array_f64(roots);
        return EXIT_FAILURE;
    }

    printf("x1 = %g\n", ((double *)roots->data)[0]);
    printf("x2 = %g\n", ((double *)roots->data)[1]);

    oif_free_array_f64(roots);

    return 0;
}
