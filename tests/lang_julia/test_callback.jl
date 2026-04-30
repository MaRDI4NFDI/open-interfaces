using Test

using OpenInterfaces
using OpenInterfacesImpl.CallbackWrapper
using Libdl

@testset "check that callback wrapper can be dynamic" begin
    lib = Libdl.dlopen("liboif_aux_funcs.so")
    add_two_ints_fn = Libdl.dlsym(lib, :add_two_ints)
    add_two_doubles_fn = Libdl.dlsym(lib, :add_two_doubles)
    add_i32_f64__ret_i32_fn = Libdl.dlsym(lib, :add_i32_f64__ret_i32)
    add_i32_f64__ret_f64_fn = Libdl.dlsym(lib, :add_i32_f64__ret_f64)

    w_f64_f64__ret_f64 = make_wrapper_over_c_callback(add_two_doubles_fn, (OIF_TYPE_F64, OIF_TYPE_F64), OIF_TYPE_F64)
    @test w_f64_f64__ret_f64((1.4, 2.7)) ≈ 4.1

    w_i32_i32__ret_i32 = make_wrapper_over_c_callback(add_two_ints_fn, (OIF_TYPE_I32, OIF_TYPE_I32), OIF_TYPE_I32)
    @test w_i32_i32__ret_i32((42, 69)) == 111

    add_i32_f64__ret_i32_wrapper = make_wrapper_over_c_callback(
        add_i32_f64__ret_i32_fn, (OIF_TYPE_I32, OIF_TYPE_F64), OIF_TYPE_I32
    )
    @test add_i32_f64__ret_i32_wrapper((12, 3.14)) == 15

    add_i32_f64__ret_f64_wrapper = make_wrapper_over_c_callback(
        add_i32_f64__ret_i32_fn, (OIF_TYPE_I32, OIF_TYPE_F64), OIF_TYPE_F64
    )
    @test add_i32_f64__ret_f64_wrapper((12, 3.14)) ≈ 15.14
end
