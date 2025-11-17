# def _parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument(
#         "impl",
#         choices=["c_qeq_solver", "py_qeq_solver", "jl_qeq_solver"],
#         default="c_qeq_solver",
#         nargs="?",
#     )
#     args = p.parse_args()
#     return Args(**vars(args))

using Printf

using OpenInterfaces
using OpenInterfaces.Interfaces.QEQ


function parse_args(args)
    supported_impls = ["c_qeq_solver", "jl_qeq_solver", "py_qeq_solver"]
    if length(args) > 0
        impl = args[1]

        if ! (impl in supported_impls)
            error("""
                  Given implementation $impl is not in the list\
                  of supported implmenetations: $supported_impls""")
        end
    else
        impl = "jl_qeq_solver"
    end

    return impl
end


function main(args)
    impl = parse_args(args)
    println("Calling from Julia an open interface for quadratic solver")
    println("Implementation: ", impl)

    implh = load_impl("qeq", impl, 1, 0)
    a, b, c = 1.0, 5.0, 4.0
    roots = QEQ.solve_qeq(implh, a, b, c)

    println("Solving quadratic equation for a = $a, b = $b, c = $c")
    println("x = ", roots)
end


main(ARGS)
