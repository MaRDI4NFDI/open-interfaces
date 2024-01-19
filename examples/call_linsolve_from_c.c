#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <oif/api.h>
#include <oif/c_bindings.h>
#include <oif/interfaces/linsolve.h>

char *parse_impl(int argc, char *argv[]) {
    if (argc == 1) {
        return "c_lapack";
    } else {
        if ((strcmp(argv[1], "c_lapack") == 0) ||
            (strcmp(argv[1], "numpy") == 0)) {
            return argv[1];
        } else {
            fprintf(stderr, "USAGE: %s [c_lapack|numpy]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char *argv[]) {
    char *impl = parse_impl(argc, argv);
    printf("Calling from C an open interface for solving Ax = b \n");
    printf("Implementation: %s\n", impl);

    ImplHandle implh = oif_init_impl("linsolve", impl, 1, 0);
    if (implh == OIF_IMPL_INIT_ERROR) {
        fprintf(stderr,
                "Error during implementation initialization. Cannot proceed\n");
        return EXIT_FAILURE;
    }

    OIFArrayF64 *A = oif_init_array_f64_from_data(
        2, (intptr_t[2]){2, 2}, (double[4]){1.0, 1.0, -3.0, 1.0});
    OIFArrayF64 *b = oif_init_array_f64_from_data(
        1, (intptr_t[1]){2}, (double[2]){6.0, 2.0});

    OIFArrayF64 *roots = oif_create_array_f64(1, (intptr_t[]){2});
    int status = oif_solve_linear_system(implh, A, b, roots);
    if (status) {
        oif_free_array_f64(roots);
        oif_free_array_f64(A);
        oif_free_array_f64(b);
        return EXIT_FAILURE;
    }

    printf("Solving Ax = b\n");
    printf("A = ");
    oif_print_matrix(A);
    printf("b = ");
    oif_print_vector(b);

    printf("Solution: ");
    oif_print_vector(roots);

    oif_free_array_f64(roots);
    oif_free_array_f64(A);
    oif_free_array_f64(b);

    return 0;
}
