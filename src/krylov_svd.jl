using KrylovKit

function krylov_svd(A, nev=1; tol=1e-8)
    S, U, V, info = svdsolve(A, nev; tol)
    stack(U; dims=2), S, stack(V, dims=2)
end
