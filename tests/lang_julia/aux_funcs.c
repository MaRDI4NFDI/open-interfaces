#include <oif/api.h>

typedef struct context {
    int p1;
    double p2;
} Context;

int
add_two_ints(int a, int b)
{
    return a + b;
}

double
add_two_doubles(double a, double b)
{
    return a + b;
}

int
add_i32_f64__ret_i32(int a, double b)
{
    return (int)(a + b);
}

double
add_i32_f64__ret_f64(int a, double b)
{
    return a + b;
}

int
axpy_f64_arrf64_arrf64__ret_i32(double alpha, OIFArrayF64 *x, OIFArrayF64 *y)
{
    int N = x->dimensions[0];
    for (int i = 0; i < N; i++) {
        y->data[i] = y->data[i] + alpha * x->data[i];
    }

    return 0;
}

int
axpy_with_context_f64_arrf64_arrf64__ret_i32(double alpha, OIFArrayF64 *x, OIFArrayF64 *y,
                                             void *void_context)
{
    Context *context = void_context;
    int N = x->dimensions[0];
    for (int i = 0; i < N; i++) {
        y->data[i] = y->data[i] + (alpha + context->p2) * (x->data[i] + context->p1);
    }

    return 0;
}
