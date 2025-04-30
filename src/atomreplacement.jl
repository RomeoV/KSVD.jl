function fasterror!(E, Y, D, X; timer::TimerOutput=TimerOutput())
    @timeit_debug timer "compute fast error" begin
        E .= copy(Y)
        fastdensesparsemul_threaded!(E, D, X, -1, 1)
        return E
    end
end
