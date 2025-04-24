"""
    fasterror(Y, D, X; timer::TimerOutput=TimerOutput())

See `fasterror!`.
"""
fasterror(Y, D, X; timer::TimerOutput=TimerOutput()) = fasterror!(similar(Y), Y, D, X; timer)

"""
    fasterror!(E, Y, D, X; timer::TimerOutput=TimerOutput())

Computes the error `E = Y - D * X`. This function is optimized to only compute the
columns of E for which the corresponding column of X is non-zero.
"""
function fasterror!(E, Y, D, X; timer::TimerOutput=TimerOutput())
    @timeit_debug timer "compute fast error" begin
        E .= copy(Y)
        fastdensesparsemul_threaded!(E, D, X, -1, 1)
        return E
    end
end
