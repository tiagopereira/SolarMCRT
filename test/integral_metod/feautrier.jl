include("shift_tools.jl")
using FastGaussQuadrature

"""
1.5D feautrier calculation.
"""
function feautrier(S, χ, z, nμ::Int64, nφ::Int64, pixel_size)

    # ===================================================================
    # SET UP VARIABLES
    # ===================================================================
    μ, w = gausslegendre(nμ)
    μ = μ ./2.0 .+ 0.5
    nz, nx, ny = size(χ)

    D = Array{Float64,3}(undef,nz-1,nx,ny)
    E = Array{Float64,3}(undef,nz-1,nx,ny)u"kW / m^2 / sr / nm"
    p = zeros(nz,nx,ny)u"kW / m^2 / sr / nm"
    P = zeros(nz,nx,ny)u"kW / m^2 / sr / nm"
    J = zeros(nz,nx,ny)u"kW / m^2 / sr / nm"

    # ==================================================================
    # FEAUTRIER
    # ==================================================================
    for m=1:nμ

        S_ = copy(S)
        χ_ = copy(χ)

        println(m)

        # Shift atmosphere
        #shift_variable!(S_, z[1:end-1], pixel_size, μ[m])
        #shift_variable!(χ_, z[1:end-1], pixel_size, μ[m])

        χ_ /= μ[m]
        S_ /= μ[m]

        # ϕ = 0
        τ = optical_depth(χ_, z)
        p[end,:,:] = forward(D, E, S_, τ, μ[m])
        backward(p, D, E)
        #P += p

        # ϕ = π
        """S_ = reverse(S_, dims = 2)
        S_ = reverse(S_, dims = 3)
        χ_ = reverse(χ_, dims = 2)
        χ_ = reverse(χ_, dims = 3)

        τ = optical_depth(χ_, z)
        p[end,:,:] = forward(D, E, S_, τ, μ[m])
        backward(p, D, E)
        p = reverse(p, dims = 2)
        p = reverse(p, dims = 3)
        P += p

        # ϕ = 3π/2
        S_ = permutedims(S_, [1,3,2])
        S_ = reverse(S_, dims = 2)
        χ_ = permutedims(χ_, [1,3,2])
        χ_ = reverse(χ_, dims = 2)

        τ = optical_depth(χ_, z)
        p[end,:,:] = forward(D, E, S_, τ, μ[m])
        backward(p, D, E)
        p = permutedims(p, [1,3,2])
        p = reverse(p, dims = 3)
        P += p

        # ϕ = π/2
        reverse(S_, dims = 2)
        reverse(S_, dims = 3)
        reverse(χ_, dims = 2)
        reverse(χ_, dims = 3)

        τ = optical_depth(χ_, z)
        p[end,:,:] = forward(D, E, S_, τ, μ[m])
        backward(p, D, E)
        reverse(p, dims = 2)
        p = permutedims(p, [1,3,2])
        p = reverse(p, dims = 2)
        P += p"""

        # Shift back μ
        #shift_variable!(P, z[1:end-1], pixel_size, -μ[m])
        #shift_variable!(P, z[1:end-1], pixel_size, -1.0)

        # Add to J
        J = J .+ w[m]*p #/nφ
    end

    return p# J
end

"""
Forward-propagation to find the coefficients.
"""
function forward(D::Array{Float64, 3},
                 E::Array{<:Unitful.Quantity,3},
                 S::Array{<:Unitful.Quantity,3},
                 τ::Array{Float64, 3},
                 μ::Float64)

    nz, nx, ny =  size(τ)
    P_end = Array{Float64,2}(undef,nx,ny)u"kW / m^2 / sr / nm"

    for j=1:ny
        for i=1:nx
            Δτ = τ[2:end,i,j] .- τ[1:end-1,i,j]

            # From boundary condition at the top
            E[1,i,j] = 0.0u"kW / m^2 / sr / nm"
            D[1,i,j] = 1.0/(Δτ[1]/μ + 1.0)

            #forward
            for k=2:length(Δτ)
                A = 2μ^2 / (Δτ[k-1]*(Δτ[k-1] + Δτ[k]))
                B = 1.0 + 2μ^2 / (Δτ[k]*Δτ[k-1])            #should use steins trick here
                C = 2μ^2 /(Δτ[k]*(Δτ[k-1] + Δτ[k]))

                D[i] = C / (B - A*D[k-1,i,j])
                E[i] = (S[k,i,j] + A*E[k-1,i,j]) / (B - A*D[k-1,i,j])
            end

            # From boundary condition at the bottom
            P_end[i,j] = ( E[end,i,j] + Δτ[end]/μ * S[end,i,j] + (S[end,i,j] - S[end-1,i,j]) ) / (1.0 - D[end,i,j] + Δτ[end]/μ)
        end
    end

    return P_end
end

"""
Back-propagation to find the P.
"""
function backward(P::Array{<:Unitful.Quantity,3},
                  D::Array{Float64, 3},
                  E::Array{<:Unitful.Quantity,3})

    nz, nx, ny = size(D)

    for j=1:ny
        for i=1:nx
            for k in range(nz, step=-1,stop=1)
                P[k,i,j] = D[k,i,j]*P[k+1,i,j] + E[k,i,j]
            end
        end
    end
end
