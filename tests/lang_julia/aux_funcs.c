int add_two_ints(int a, int b) {
    return a + b;
}

double add_two_doubles(double a, double b) {
    return a + b;
}

int add_i32_f64__ret_i32(int a, double b) {
    return (int) (a + b);
}

double add_i32_f64__ret_f64(int a, double b) {
    return a + b;
}
