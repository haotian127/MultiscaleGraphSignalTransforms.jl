"""
    function LPHGLET_Synthesis(dvec::Vector{Float64}, GP::GraphPart, BS::BasisSpec, G::GraphSig; gltype::Symbol = :L, ϵ::Float64 = 0.3)

Perform Lapped-HGLET Synthesis transform

### Input Arguments
* `dvec`: the expansion coefficients corresponding to the chosen basis
* `GP`: a GraphPart object
* `BS`: a BasisSpec object
* `G`: a GraphSig object
* `gltype`: :L or :Lsym, indicating which eigenvectors are used
* `ϵ`: relative action bandwidth (default: 0.3)

### Output Argument
* `f`: the reconstructed signal
* `GS`: the reconstructed GraphSig object
"""
function LPHGLET_Synthesis(dvec::Matrix{Float64}, GP::GraphPart, BS::BasisSpec, G::GraphSig; gltype::Symbol = :L, ϵ::Float64 = 0.3)
    # Preliminaries
    W = G.W
    inds = GP.inds
    rs = GP.rs
    N = size(W, 1)
    jmax = size(rs, 2)
    Uf = Matrix{Float64}(I, N, N)
    used_node = Set()

    # fill in the appropriate entries of dmatrix
    dmatrix = dvec2dmatrix(dvec, GP, BS)

    f = zeros(size(dmatrix[:, jmax, :]))


    # Perform the synthesis transform
    for j = 1:jmax
        regioncount = count(!iszero, rs[:,j]) - 1
        # assemble orthogonal folding operator at level j - 1
        keep_folding!(Uf, used_node, W, GP; ϵ = ϵ, j = j - 1)
        for r = 1:regioncount
            # indices of current region
            indr = rs[r, j]:(rs[r + 1, j] - 1)
            # indices of current region's nodes
            indrs = inds[indr, j]
            # number of nodes in current region
            n = length(indrs)

            # only proceed forward if coefficients do not exist
            if (j == jmax || count(!iszero, dmatrix[indr, j + 1, :]) == 0) && count(!iszero, dmatrix[indr, j, :]) > 0
                # compute the eigenvectors
                W_temp = W[indrs,indrs]
                D_temp = Diagonal(vec(sum(W_temp, dims = 1)))
                if gltype == :L
                    # compute the eigenvectors of L ==> svd(L)
                    v = svd(Matrix(D_temp - W_temp)).U
                elseif gltype == :Lsym
                    # check if one can assemble the Lsym
                    if minimum(sum(W[indrs, indrs], dims = 1)) > 10^3 * eps()
                        ### eigenvectors of L_sym ==> svd(L_sym)
                        D_temp_p = Diagonal(vec(sum(W_temp, dims = 1)).^(-0.5))
                        v = svd(Matrix(D_temp_p * (D_temp - W_temp) * D_temp_p)).U
                    else
                        ### eigenvectors of L ==> svd(L)
                        v = svd(Matrix(D_temp - W_temp)).U
                    end
                end
                v = v[:,end:-1:1]


                # standardize the eigenvector signs
                standardize_eigenvector_signs!(v)

                # construct unfolder operator custom to current region
                P = Uf[indrs, :]'

                # reconstruct the signal
                f += (P * v) * dmatrix[indr, j, :]

            end
        end
    end

    # creat a GraphSig object with the reconstructed data
    GS = deepcopy(G)
    replace_data!(GS, f)

    return f, GS
end



"""
    function LPHGLET_Analysis_All(G::GraphSig, GP::GraphPart; ϵ::Float64 = 0.3)

For a GraphSig object 'G', generate the 2 matrices of Lapped-HGLET expansion coefficients
corresponding to the eigenvectors of L and Lsym

### Input Arguments
* `G`:  a GraphSig object
* `GP`: a GraphPart object
* `ϵ`: relative action bandwidth (default: 0.3)

### Output Argument
* `dmatrixlH`:        the matrix of expansion coefficients for L
* `dmatrixlHsym`:     the matrix of expansion coefficients for Lsym
* `GP`:              a GraphPart object
"""
function LPHGLET_Analysis_All(G::GraphSig, GP::GraphPart; ϵ::Float64 = 0.3)
    # Preliminaries
    W = G.W
    inds = GP.inds
    rs = GP.rs
    N = size(W, 1)
    jmax = size(rs, 2)
    fcols = size(G.f, 2)
    Uf = Matrix{Float64}(I, N, N)
    used_node = Set()
    dmatrixlH = zeros(N, jmax, fcols)
    dmatrixlHsym = deepcopy(dmatrixlH)

    for j = 1:jmax
        regioncount = count(!iszero, rs[:,j]) - 1
        # assemble orthogonal folding operator at level j - 1
        keep_folding!(Uf, used_node, W, GP; ϵ = ϵ, j = j - 1)
        for r = 1:regioncount
            # indices of current region
            indr = rs[r, j]:(rs[r + 1, j] - 1)
            # indices of current region's nodes
            indrs = inds[indr, j]
            # number of nodes in current region
            n = length(indrs)

            # compute the eigenvectors
            W_temp = W[indrs,indrs]
            D_temp = Diagonal(vec(sum(W_temp, dims = 1)))
            ## eigenvectors of L ==> svd(L)
            v = svd(Matrix(D_temp - W_temp)).U
            ## eigenvectors of L_sym ==> svd(L_sym)
            if minimum(sum(W[indrs, indrs], dims = 1)) > 10^3 * eps()
                ### eigenvectors of L_sym ==> svd(L_sym)
                D_temp_p = Diagonal(vec(sum(W_temp, dims = 1)).^(-0.5))
                v_sym = svd(Matrix(D_temp_p * (D_temp - W_temp) * D_temp_p)).U
            else
                ### eigenvectors of L ==> svd(L)
                v_sym = deepcopy(v)
            end

            # standardize the eigenvector signs
            v = v[:,end:-1:1]
            standardize_eigenvector_signs!(v)
            v_sym = v_sym[:,end:-1:1]
            standardize_eigenvector_signs!(v_sym)

            # construct unfolding operator custom to current region
            P = Uf[indrs, :]'
            # obtain the expansion coefficients
            dmatrixlH[indr, j, :] = (P * v)' * G.f
            dmatrixlHsym[indr, j, :] = (P * v_sym)' * G.f
        end
    end

    return dmatrixlH, dmatrixlHsym

end



function standardize_eigenvector_signs!(v)
    # standardize the eigenvector signs for HGLET (different with NGWPs)
    for col = 1:size(v, 2)
        row = 1
        standardized = false
        while !standardized
            if v[row, col] > 10^3 * eps()
                standardized = true
            elseif v[row,col] < -10^3 * eps()
                v[:, col] = -v[:, col]
            else
                row += 1
            end
        end
    end
end

"""
    HGLET_dictionary(GP::GraphPart, G::GraphSig; gltype::Symbol = :L)

assemble the whole HGLET dictionary

### Input Arguments
* `GP`: a GraphPart object
* `G`:  a GraphSig object
* `gltype`: `:L` or `:Lsym`

### Output Argument
* `dictionary`: the HGLET dictionary

"""
function HGLET_dictionary(GP::GraphPart, G::GraphSig; gltype::Symbol = :L)
    N = size(G.W, 1)
    jmax = size(GP.rs, 2)
    dictionary = zeros(N, jmax, N)
    for j = 1:jmax
        BS = BasisSpec(collect(enumerate(j * ones(Int, N))))
        dictionary[:, j, :] = HGLET_Synthesis(Matrix{Float64}(I, N, N), GP, BS, G; gltype = gltype)[1]'
    end
    return dictionary
end

"""
    LPHGLET_dictionary(GP::GraphPart, G::GraphSig; gltype::Symbol = :L, ϵ::Float64 = 0.3)

assemble the whole LP-HGLET dictionary

### Input Arguments
* `GP`: a GraphPart object
* `G`:  a GraphSig object
* `gltype`: `:L` or `:Lsym`
* `ϵ`: relative action bandwidth (default: 0.3)

### Output Argument
* `dictionary`: the LP-HGLET dictionary

"""
function LPHGLET_dictionary(GP::GraphPart, G::GraphSig; gltype::Symbol = :L, ϵ::Float64 = 0.3)
    N = size(G.W, 1)
    jmax = size(GP.rs, 2)
    dictionary = zeros(N, jmax, N)
    for j = 1:jmax
        BS = BasisSpec(collect(enumerate(j * ones(Int, N))))
        dictionary[:, j, :] = LPHGLET_Synthesis(Matrix{Float64}(I, N, N), GP, BS, G; gltype = gltype, ϵ = ϵ)[1]'
    end
    return dictionary
end


function HGLET_DST4_Analysis(G::GraphSig, GP::GraphPart)
    # Preliminaries
    W = G.W
    ind = GP.ind
    rs = GP.rs
    N = size(G.W,1)
    jmax = size(rs,2)
    fcols = size(G.f,2)
    dmatrix = zeros(N,jmax,fcols)
    dmatrix[:,jmax,:] = G.f[ind,:]

    # Perform the HGLET analysis, i.e., generating the HGLET coefficients
    for j = jmax-1:-1:1
        regioncount = count(!iszero, rs[:,j]) - 1
        for r = 1:regioncount
            # the index that marks the start of the region
            rs1 = rs[r,j]

            # the index that is one after the end of the region
            rs3 = rs[r+1,j]

            # the number of points in the current region
            n = rs3 - rs1

            if n == 1
                dmatrix[rs1,j,:] = G.f[ind[rs1],:]
            elseif n > 1
                indrs = ind[rs1:rs3-1]
                W_temp = W[indrs,indrs]
                D_temp = Diagonal(vec(sum(W_temp, dims = 1)))
                L_temp = D_temp - W_temp
                L_temp[1, 1] = 3  # DST type-IV (left: Dirichlet, right: Neumann)
                v = svd(Matrix(L_temp)).U
                v = v[:,end:-1:1] # reorder the ev's in the decreasing ew's

                # standardize the eigenvector signs
                for col = 1:n
                    row = 1
                    standardized = false
                    while !standardized
                        if v[row,col] > 10^3*eps()
                            standardized = true
                        elseif v[row,col] < -10^3*eps()
                            v[:,col] = - v[:,col]
                            standardized = true
                        else
                            row += 1
                        end
                    end
                end

                # obtain the expansion coeffcients
                if gltype == :Lrw && normalizep
                    dmatrix[rs1:rs3-1,j,:] = v'*(D.^0.5)*G.f[indrs,:]
                else
                    dmatrix[rs1:rs3-1,j,:] = v'*G.f[indrs,:]
                end
            end
        end
    end
    return dmatrix
end


function HGLET_DST4_Synthesis(dvec::Matrix{Float64}, GP::GraphPart, BS::BasisSpec,
                         G::GraphSig)
    # Preliminaries
    W = G.W
    jmax = size(GP.rs,2)

    # Fill in the appropriate entries of dmatrix
    dmatrix = dvec2dmatrix(dvec,GP,BS)
    f = dmatrix[:,jmax,:]

    # Perform the signal synthesis from the given coefficients
    for j = jmax:-1:1
        regioncount = count(!iszero, GP.rs[:,j]) - 1
        for r = 1:regioncount
            # the index that marks the start of the region
            rs1 = GP.rs[r,j]

            # the index that is one after the end of the region
            rs3 = GP.rs[r+1,j]

            # the number of points in the current region
            n = rs3 - rs1

            # only proceed forward if coefficients do not exist
            if (j == jmax || count(!iszero, dmatrix[rs1:rs3-1,j+1,:]) == 0) && count(!iszero, dmatrix[rs1:rs3-1,j,:]) > 0

                if n == 1
                    f[rs1,:] = dmatrix[rs1,j,:]
                elseif n > 1
                    indrs = GP.ind[rs1:rs3-1]
                    W_temp = W[indrs,indrs]
                    D_temp = Diagonal(vec(sum(W_temp, dims = 1)))
                    L_temp = D_temp - W_temp
                    L_temp[1, 1] = 3  # DST type-IV (left: Dirichlet, right: Neumann)
                    v = svd(Matrix(L_temp)).U
                    v = v[:,end:-1:1] # reorder the ev's in the decreasing ew's

                    # standardize the eigenvector signs
                    standardize_eigenvector_signs!(v)

                    # reconstruct the signal
                    f[rs1:rs3-1,:] = v*dmatrix[rs1:rs3-1,j,:]
                end
            end
        end
    end

    # put the reconstructed values in the correct order
    f[GP.ind,:] = f

    # creat a GraphSig object with the reconstructed data
    GS = deepcopy(G)
    replace_data!(GS,f)

    return f, GS
end


function HGLET_DST4_dictionary(GP::GraphPart, G::GraphSig)
    N = size(G.W, 1)
    jmax = size(GP.rs, 2)
    dictionary = zeros(N, jmax, N)
    for j = 1:jmax
        BS = BasisSpec(collect(enumerate(j * ones(Int, N))))
        dictionary[:, j, :] = HGLET_DST4_Synthesis(Matrix{Float64}(I, N, N), GP, BS, G)[1]'
    end
    return dictionary
end


function LPHGLET_DST4_Analysis(G::GraphSig, GP::GraphPart; ϵ::Float64 = 0.3)
    # Preliminaries
    W = G.W
    inds = GP.inds
    rs = GP.rs
    N = size(W, 1)
    jmax = size(rs, 2)
    fcols = size(G.f, 2)
    Uf = Matrix{Float64}(I, N, N)
    used_node = Set()
    dmatrixlH = zeros(N, jmax, fcols)

    for j = 1:jmax
        regioncount = count(!iszero, rs[:,j]) - 1
        # assemble orthogonal folding operator at level j - 1
        keep_folding!(Uf, used_node, W, GP; ϵ = ϵ, j = j - 1)
        for r = 1:regioncount
            # indices of current region
            indr = rs[r, j]:(rs[r + 1, j] - 1)
            # indices of current region's nodes
            indrs = inds[indr, j]
            # number of nodes in current region
            n = length(indrs)

            # compute the eigenvectors
            W_temp = W[indrs,indrs]
            D_temp = Diagonal(vec(sum(W_temp, dims = 1)))
            L_temp = D_temp - W_temp
            L_temp[1, 1] = 3  # DST type-IV (left: Dirichlet, right: Neumann)
            v = svd(Matrix(L_temp)).U

            # standardize the eigenvector signs
            v = v[:,end:-1:1]
            standardize_eigenvector_signs!(v)

            # construct unfolding operator custom to current region
            P = Uf[indrs, :]'
            # obtain the expansion coefficients
            dmatrixlH[indr, j, :] = (P * v)' * G.f
        end
    end

    return dmatrixlH

end


function LPHGLET_DST4_Synthesis(dvec::Matrix{Float64}, GP::GraphPart, BS::BasisSpec, G::GraphSig; ϵ::Float64 = 0.3)
    # Preliminaries
    W = G.W
    inds = GP.inds
    rs = GP.rs
    N = size(W, 1)
    jmax = size(rs, 2)
    Uf = Matrix{Float64}(I, N, N)
    used_node = Set()

    # fill in the appropriate entries of dmatrix
    dmatrix = dvec2dmatrix(dvec, GP, BS)
    f = zeros(size(dmatrix[:, jmax, :]))

    # Perform the synthesis transform
    for j = 1:jmax
        regioncount = count(!iszero, rs[:,j]) - 1
        # assemble orthogonal folding operator at level j - 1
        MultiscaleGraphSignalTransforms.keep_folding!(Uf, used_node, W, GP; ϵ = ϵ, j = j - 1)
        for r = 1:regioncount
            # indices of current region
            indr = rs[r, j]:(rs[r + 1, j] - 1)
            # indices of current region's nodes
            indrs = inds[indr, j]
            # number of nodes in current region
            n = length(indrs)

            # only proceed forward if coefficients do not exist
            if (j == jmax || count(!iszero, dmatrix[indr, j + 1, :]) == 0) && count(!iszero, dmatrix[indr, j, :]) > 0
                # compute the eigenvectors
                W_temp = W[indrs,indrs]
                D_temp = Diagonal(vec(sum(W_temp, dims = 1)))
                L_temp = D_temp - W_temp
                L_temp[1, 1] = 3  # DST type-IV (left: Dirichlet, right: Neumann)
                v = svd(Matrix(L_temp)).U
                v = v[:,end:-1:1]

                # standardize the eigenvector signs
                standardize_eigenvector_signs!(v)

                # construct unfolder operator custom to current region
                P = Uf[indrs, :]'

                # reconstruct the signal
                f += (P * v) * dmatrix[indr, j, :]

            end
        end
    end

    # creat a GraphSig object with the reconstructed data
    GS = deepcopy(G)
    replace_data!(GS, f)

    return f, GS
end


function LPHGLET_DST4_dictionary(GP::GraphPart, G::GraphSig; ϵ::Float64 = 0.3)
    N = size(G.W, 1)
    jmax = size(GP.rs, 2)
    dictionary = zeros(N, jmax, N)
    for j = 1:jmax
        BS = BasisSpec(collect(enumerate(j * ones(Int, N))))
        dictionary[:, j, :] = LPHGLET_DST4_Synthesis(Matrix{Float64}(I, N, N), GP, BS, G; ϵ = ϵ)[1]'
    end
    return dictionary
end
