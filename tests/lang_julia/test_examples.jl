using Test

const EXAMPLES_PATH = joinpath("examples", "lang_julia")
const CALL_QEQ = joinpath(EXAMPLES_PATH, "call_qeq.jl")
const CALL_LINSOLVE = joinpath(EXAMPLES_PATH, "call_linsolve.jl")
const CALL_IVP = joinpath(EXAMPLES_PATH, "call_ivp.jl")
const CALL_IVP_BURGERS = joinpath(EXAMPLES_PATH, "call_ivp_burgers_eq.jl")
const CALL_IVP_VDP = joinpath(EXAMPLES_PATH, "call_ivp_vdp.jl")

@testset "Testing Julia examples" begin
    @testset "Testing QEQ example with all implementations" begin
        for impl in ["c_qeq_solver", "jl_qeq_solver", "py_qeq_solver"]
            @testset "call_qeq.jl $impl runs successfully" begin
                process = run(
                    pipeline(`julia $CALL_QEQ $impl`, stdout = devnull, stderr = devnull),
                )
                @test process.exitcode == 0
            end
        end

        @testset "call_qeq.jl without args runs successfully" begin
            process = run(pipeline(`julia $CALL_QEQ`, stdout = devnull, stderr = devnull))
            @test process.exitcode == 0
        end
    end


    @testset "Testing LINSOLVE example with all implementations" begin
        for impl in ["c_lapack", "jl_backslash", "numpy"]
            @testset "call_linsolve.jl $impl runs successfully" begin
                process = run(
                    pipeline(
                        `julia $CALL_LINSOLVE $impl`,
                        stdout = devnull,
                        stderr = devnull,
                    ),
                )
                @test process.exitcode == 0
            end
        end

        @testset "call_linsolve.jl without args runs successfully" begin
            process =
                run(pipeline(`julia $CALL_LINSOLVE`, stdout = devnull, stderr = devnull))
            @test process.exitcode == 0
        end
    end


    @testset "Testing IVP example with all implementations" begin
        for impl in ["sundials_cvode", "jl_diffeq", "scipy_ode"]
            @testset "call_ivp.jl $impl runs successfully" begin
                process = run(
                    pipeline(`julia $CALL_IVP $impl`, stdout = devnull, stderr = devnull),
                )
                @test process.exitcode == 0
            end
        end

        @testset "call_ivp.jl without args runs successfully" begin
            process = run(pipeline(`julia $CALL_IVP`, stdout = devnull, stderr = devnull))
            @test process.exitcode == 0
        end
    end


    @testset "Testing IVP Burgers example with all implementations" begin
        for impl in ["sundials_cvode", "jl_diffeq", "scipy_ode"]
            @testset "call_ivp_burgers_eq.jl $impl runs successfully" begin
                process = run(
                    pipeline(
                        `julia $CALL_IVP_BURGERS $impl --no-plot`,
                        stdout = devnull,
                        stderr = devnull,
                    ),
                )
                @test process.exitcode == 0
            end
        end

        @testset "call_ivp_burgers_eq.jl without args runs successfully" begin
            process = run(
                pipeline(
                    `julia $CALL_IVP_BURGERS --no-plot`,
                    stdout = devnull,
                    stderr = devnull,
                ),
            )
            @test process.exitcode == 0
        end
    end

    # --------------------------------------------------------------------------
    # BEGIN Van der Pol equation: unsuccessful and successful runs.
    @testset "Testing IVP VdP example with all implementations" begin
        for impl in ["scipy_ode-dopri5", "scipy_ode-dopri5-100k", "scipy_ode-vode"]
            @testset "call_ivp_vdp.jl $impl fails due to numerics" begin
                io_err = IOBuffer()
                process = try
                    run(
                        pipeline(
                            `julia $CALL_IVP_VDP $impl --no-plot`,
                            stdout = devnull,
                            stderr = io_err,
                        ),
                    )
                catch e
                end
                captured_stderr = String(take!(io_err))
                @test occursin("Call to the method 'integrate' has failed", captured_stderr)
            end
        end

        for impl in [
            "scipy_ode-vode-40k",
            # "sundials_cvode-default",  # It is simply too slow to run (40 secs)
            "jl_diffeq-rosenbrock23",
        ]
            @testset "call_ivp_vdp.jl $impl succeeds" begin
                io_err = IOBuffer()
                process = try
                    run(pipeline(`julia $CALL_IVP_VDP $impl --no-plot`, stderr = io_err))
                catch e
                end
                captured_stderr = String(take!(io_err))
                println("Captured stderr:")
                println("--- BEGIN")
                println(captured_stderr)
                println("--- END")
                @test process.exitcode == 0
            end
        end
    end
    # END Van der Pol equation: unsuccessful and successful runs.
    # --------------------------------------------------------------------------
end
