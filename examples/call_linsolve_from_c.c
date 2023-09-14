#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oif/api.h"
#include <oif/frontend_c/frontend_c.h>
#include <oif/frontend_c/linsolve.h>


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
    printf("Calling from C an open interface for solving Ax = b \n");
    printf("Backend: %s\n", backend);

    OIFArrayF64 *A = init_array_f64_from_data(
        2, (intptr_t [2]){2, 2}, (double [4]){1.0, 1.0, -3.0, 1.0}
    );
    OIFArrayF64 *b = init_array_f64_from_data(
        1, (intptr_t [1]){2}, (double [2]){6.0, 2.0}
    );

    BackendHandle bh = oif_init_backend(backend, "", 1, 0);
    if (bh == OIF_BACKEND_INIT_ERROR) {
        fprintf(stderr, "Error during backend initialization. Cannot proceed\n");
        return EXIT_FAILURE;
    }

    OIFArrayF64 *roots = create_array_f64(1, (intptr_t []){2});
    int status = oif_solve_linear_system(bh, A, b, roots);
    if (status) {
        free_array_f64(roots);
        free_array_f64(A);
        free_array_f64(b);
        return EXIT_FAILURE;
    }

    printf("Solving Ax = b\n");
    printf("A = ");
    oif_print_matrix(A);
    printf("b = ");
    oif_print_vector(b);

    printf("Solution: ");
    oif_print_vector(roots);

    free_array_f64(roots);
    free_array_f64(A);
    free_array_f64(b);

    return 0;
}
