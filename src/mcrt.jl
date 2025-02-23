include("radiation.jl")


"""
     mcrt(atmosphere::Atmosphere,
          radiation::Radiation,
          atom::Atom,
          lines,
          lineRadiations,
          max_scatterings::Real,
          iteration::Int64,
          output_path::String)

Monte Carlo radiative transfer simulation
for full population-iteration mode."""
function mcrt(atmosphere::Atmosphere,
              radiation::Radiation,
              atom::Atom,
              lines,
              lineRadiations,
              max_scatterings::Real,
              iteration::Int64,
              output_path::String)

    # ==================================================================
    # ATMOSPHERE DATA
    # ==================================================================
    x = atmosphere.x
    y = atmosphere.y
    z = atmosphere.z
    velocity = atmosphere.velocity

    # ===================================================================
    # RADIATION DATA
    # ===================================================================
    α_continuum = radiation.α_continuum
    ε_continuum = radiation.ε_continuum
    boundary = radiation.boundary
    packets = radiation.packets

    nλ, nz, nx, ny = size(packets)

    # ===================================================================
    # ATOM DATA
    # ===================================================================
    iλbb = atom.iλbb
    λ = atom.λ
    n_lines = atom.n_lines
    nλ = atom.nλ

    # ===================================================================
    # SIMULATION
    # ===================================================================
    println(@sprintf("--Starting simulation, using %d thread(s)...",
            Threads.nthreads()))

    # Initialise placeholder variable
    J_λ = Array{Float64,3}(undef, nz, nx, ny)
    I0_λ = Array{Int64,3}(undef, 3, nx, ny)

    stop = nothing

    # Go through all lines and the bound-free in-between
    for ln=1:n_lines

        start,stop = iλbb[ln]

        # Bound-free
        for λi=1:start-1

            # Reset counters
            fill!(J_λ, 0.0)
            fill!(I0_λ, 0.0)
            total_destroyed = Threads.Atomic{Int64}(0)
            total_scatterings = Threads.Atomic{Int64}(0)

            # Pick out wavelength data
            packets_λ = packets[λi,:,:,:]
            J_λ .= packets_λ
            α_λ = α_continuum[λi,:,:,:]
            boundary_λ = boundary[λi,:,:]
            ε_λ = ε_continuum[λi,:,:,:]

            println("\n--[", λi,"/",nλ, "]        ", @sprintf("λ = %.3f nm", ustrip(λ[λi])))

            # Create ProgressMeter working with threads
            p = Progress(ny); update!(p,0)
            jj = Threads.Atomic{Int}(0)
            l = Threads.SpinLock()

            # Go through all boxes
            et = @elapsed Threads.@threads for j=1:ny

                # Advance ProgressMeter
                Threads.atomic_add!(jj, 1)
                Threads.lock(l); update!(p, jj[])
                Threads.unlock(l)

                local_destroyed = 0
                local_scattered = 0

                for i=1:nx
                    for k=1:boundary_λ[i,j]

                        # Packets in box
                        pcts = Int(packets_λ[k,i,j])

                        if pcts == 0
                            continue
                        end

                        # Dimensions of box
                        corner = [z[k], x[i], y[j]]
                        box_dim = [z[k+1], x[i+1], y[j+1]] .- corner

                        for pct=1:pcts
                            # Initial box
                            box_id = [k,i,j]

                            # Initial position uniformely drawn from box
                            r = corner .+ (box_dim .* rand(3))

                            # Scatter each packet until destroyed,
                            # escape or reach max_scatterings
                            box_id_old = box_id
                            stuck = 0

                            # Scatter each packet until destroyed,
                            # escape or reach max_scatterings
                            for s=1:Int(max_scatterings)

                                # Scatter packet once
                                box_id, r, lost = scatter_packet_continuum(x, y, z,
                                                                           α_λ,
                                                                           boundary_λ,
                                                                           box_id, r,
                                                                           J_λ, I0_λ)

                                # Check if escaped or lost in bottom
                                if lost
                                    break
                                # Check if destroyed in next particle interaction
                                elseif rand() < ε_λ[box_id...]
                                    local_destroyed += 1
                                    break
                                end

                                local_scattered += 1

                                # Check if stuck in same box
                                if box_id == box_id_old
                                    stuck += 1
                                    if stuck > 1e3
                                        break
                                    end
                                else
                                    stuck = 0
                                    box_id_old = box_id
                                end

                            end
                        end
                    end
                    for k=(boundary_λ[i,j]+1):nz
                        J_λ[k,i,j] = packets_λ[k,i,j]
                    end
                end
                Threads.atomic_add!(total_destroyed, local_destroyed)
                Threads.atomic_add!(total_scatterings, local_scattered)
            end

            # ===================================================================
            # WRITE TO FILE
            # ===================================================================
            h5open(output_path, "r+") do file
                file["J"][iteration,λi,:,:,:] = J_λ
                file["I0"][iteration,λi,:,:,:] = I0_λ
                file["total_destroyed"][iteration,λi] = total_destroyed.value
                file["total_scatterings"][iteration,λi] = total_scatterings.value
                file["total_destroyed"][iteration,λi] = total_destroyed.value
                file["time"][iteration,λi] = et
            end
        end

        line = lines[ln]
        lineRadiation = lineRadiations[ln]
        lineData = line.lineData
        dc = line.damping_constant
        ΔλD = line.doppler_width
        λ0 = lineData.λ0

        α_line_constant = lineRadiation.α_line_constant
        ε_line = lineRadiation.ε_line

        # Line
        for λi=start:stop

            # Reset counters
            fill!(J_λ, 0.0)
            fill!(I0_λ, 0.0)
            total_destroyed = Threads.Atomic{Int64}(0)
            total_scatterings = Threads.Atomic{Int64}(0)

            # Pick out wavelength data
            packets_λ = packets[λi,:,:,:]
            J_λ .= packets_λ
            boundary_λ = boundary[λi,:,:]
            α_continuum_λ = α_continuum[λi,:,:,:]
            ε_continuum_λ = ε_continuum[λi,:,:,:]

            println("\n--[",λi,"/",nλ, "]        ", @sprintf("λ = %.3f nm", ustrip(λ[λi])))

            # Create ProgressMeter working with threads
            p = Progress(ny)
            update!(p,0)
            jj = Threads.Atomic{Int}(0)
            l = Threads.SpinLock()

            # Go through all boxes
            et = @elapsed Threads.@threads for j=1:ny

                # Advance ProgressMeter
                Threads.atomic_add!(jj, 1)
                Threads.lock(l)
                update!(p, jj[])
                Threads.unlock(l)

                local_destroyed = 0
                local_scattered = 0

                for i=1:nx
                    for k=1:boundary_λ[i,j]

                        # Packets in box
                        pcts = Int(packets_λ[k,i,j])

                        if pcts == 0
                            continue
                        end

                        # Dimensions of box
                        corner = SA[z[k], x[i], y[j]]
                        box_dim = SA[z[k+1], x[i+1], y[j+1]] .- corner

                        for pct=1:pcts

                            # Initial box
                            box_id = [k,i,j]

                            # Initial position uniformely drawn from box
                            r = Vector(corner .+ (box_dim .* rand(3)))


                            # Scatter each packet until destroyed,
                            # escape or reach max_scatterings
                            box_id_old = box_id
                            stuck = 0

                            # Scatter each packet until destroyed,
                            # escape or reach max_scatterings
                            for s=1:Int(max_scatterings)

                                # Scatter packet once
                                box_id, r, lost, α = scatter_packet_line(x, y, z,
                                                                         velocity,
                                                                         α_continuum_λ,
                                                                         α_line_constant,
                                                                         boundary_λ,
                                                                         box_id, r,
                                                                         J_λ, I0_λ,
                                                                         dc, ΔλD,
                                                                         λ0, λ[λi])

                                # Check if escaped or lost in bottom
                                if lost
                                    break
                                end

                                # Check if destroyed in next particle interaction
                                α_line = α -  α_continuum_λ[box_id...]
                                ε = ( ε_continuum_λ[box_id...] * α_continuum_λ[box_id...] +
                                      ε_line[box_id...]        * α_line ) / α

                                if rand() < ε
                                    local_destroyed += 1
                                    break
                                end

                                local_scattered += 1


                                # Check if stuck in same box
                                if box_id == box_id_old
                                    stuck += 1
                                    if stuck > 1e3
                                        break
                                    end
                                else
                                    stuck = 0
                                    box_id_old = box_id
                                end

                            end
                        end
                    end
                    for k=(boundary_λ[i,j]+1):nz
                        J_λ[k,i,j] = packets_λ[k,i,j]
                    end
                end
                Threads.atomic_add!(total_destroyed, local_destroyed)
                Threads.atomic_add!(total_scatterings, local_scattered)
            end

            # ===================================================================
            # WRITE TO FILE
            # ===================================================================
            h5open(output_path, "r+") do file
                file["J"][iteration,λi,:,:,:] = J_λ
                file["I0"][iteration,λi,:,:,:] = I0_λ
                file["total_destroyed"][iteration,λi] = total_destroyed.value
                file["total_scatterings"][iteration,λi] = total_scatterings.value
                file["total_destroyed"][iteration,λi] = total_destroyed.value
                file["time"][iteration,λi] = et
            end
        end

    end

    # Final bound-free transition
    for λi=stop+1:nλ

        # Reset counters
        fill!(J_λ, 0.0)
        fill!(I0_λ, 0.0)
        total_destroyed = Threads.Atomic{Int64}(0)
        total_scatterings = Threads.Atomic{Int64}(0)

        # Pick out wavelength data
        packets_λ = packets[λi,:,:,:]
        J_λ .= packets_λ
        α_λ = α_continuum[λi,:,:,:]
        boundary_λ = boundary[λi,:,:]
        ε_λ = ε_continuum[λi,:,:,:]

        println("\n--[", λi,"/",nλ, "]        ", @sprintf("λ = %.3f nm", ustrip(λ[λi])))

        # Create ProgressMeter working with threads
        p = Progress(ny); update!(p,0)
        jj = Threads.Atomic{Int}(0)
        l = Threads.SpinLock()

        # Go through all boxes
        et = @elapsed Threads.@threads for j=1:ny

            # Advance ProgressMeter
            Threads.atomic_add!(jj, 1)
            Threads.lock(l); update!(p, jj[])
            Threads.unlock(l)

            for i=1:nx
                for k=1:boundary_λ[i,j]

                    # Packets in box
                    pcts = Int(packets_λ[k,i,j])

                    if pcts == 0
                        continue
                    end

                    # Dimensions of box
                    corner = [z[k], x[i], y[j]]
                    box_dim = [z[k+1], x[i+1], y[j+1]] .- corner

                    for pct=1:pcts
                        # Initial box
                        box_id = [k,i,j]

                        # Initial position uniformely drawn from box
                        r = corner .+ (box_dim .* rand(3))

                        # Scatter each packet until destroyed,
                        # escape or reach max_scatterings
                        box_id_old = box_id
                        stuck = 0

                        for s=1:Int(max_scatterings)

                            # Scatter packet once
                            box_id, r, lost = scatter_packet_continuum(x, y, z,
                                                                       α_λ,
                                                                       boundary_λ,
                                                                       box_id, r,
                                                                       J_λ, I0_λ)

                            # Check if escaped or lost in bottom
                            if lost
                                break
                            # Check if destroyed in next particle interaction
                            elseif rand() < ε_λ[box_id...]
                                Threads.atomic_add!(total_destroyed, 1)
                                break
                            end

                            Threads.atomic_add!(total_scatterings, 1)

                            # Check if stuck in same box
                            if box_id == box_id_old
                                stuck += 1
                                if stuck > 1e3
                                    break
                                end
                            else
                                stuck = 0
                                box_id_old = box_id
                            end

                        end
                    end
                end

                for k=(boundary_λ[i,j]+1):nz
                    J_λ[k,i,j] = packets_λ[k,i,j]
                end
            end
        end

        # ===================================================================
        # WRITE TO FILE
        # ===================================================================
        h5open(output_path, "r+") do file
            file["J"][iteration,λi,:,:,:] = J_λ
            file["I0"][iteration,λi,:,:,:] = I0_λ
            file["total_destroyed"][iteration,λi] = total_destroyed.value
            file["total_scatterings"][iteration,λi] = total_scatterings.value
            file["total_destroyed"][iteration,λi] = total_destroyed.value
            file["time"][iteration,λi] = et
        end
    end
end



"""
    scatter_packet_line(x::Array{<:Unitful.Length, 1},
                   y::Array{<:Unitful.Length, 1},
                   z::Array{<:Unitful.Length, 1},
                   velocity::Array{Array{<:Unitful.Velocity, 1}, 3},

                   α_continuum::Array{<:PerLength, 3},
                   α_line_constant::Array{Float64, 3},
                   boundary::Array{Int32, 2},

                   box_id::Array{Int64,1},
                   r::Array{<:Unitful.Length, 1},
                   J::Array{Int32, 3},
                   I0::Array{Int32, 3},

                   damping_constant::Array{<:PerArea, 3},
                   ΔλD::Array{<:Unitful.Length, 3},

                   λ0::Unitful.Length,
                   λ::Unitful.Length)

Traverse boxes until optical depth target reached
for scattering event and return new position.
"""
function scatter_packet_line(x::Array{<:Unitful.Length, 1},
                        y::Array{<:Unitful.Length, 1},
                        z::Array{<:Unitful.Length, 1},
                        velocity::Array{Array{<:Unitful.Velocity, 1}, 3},

                        α_continuum::Array{<:PerLength, 3},
                        α_line_constant::Array{Float64, 3},
                        boundary::Array{Int32, 2},

                        box_id::Array{Int64,1},
                        r::Array{<:Unitful.Length, 1},
                        J::Array{Float64, 3},
                        I0::Array{Int64, 3},

                        damping_constant::Array{<:PerArea, 3},
                        ΔλD::Array{<:Unitful.Length, 3},

                        λ0::Unitful.Length,
                        λ::Unitful.Length)

    # Keep track of status
    lost = false

    # Useful quantities
    side_dim = SVector(size(boundary))
    side_edge = SA[x[1] x[end]
                   y[1] y[end]]

    # ===================================================================
    # DRAW DEPTH AND DIRECTION
    # ===================================================================

    # Draw scattering depth and direction
    τ = -log(rand())
    ϕ = 2π * rand()
    θ =  π * rand()

    # Find direction
    unit_vector = [cos(θ), sin(θ)*cos(ϕ), sin(θ)*sin(ϕ)]
    direction = Int.(sign.(unit_vector))
    direction[1] = -direction[1] # because height array up->down

    # ===================================================================
    # MOVE PACKET TO FIRST BOX INTERSECTION
    # ===================================================================

    # Next face cross in all dimensions
    next_edge = (direction .> 0) .+ box_id

    # Closest face and distance to it
    face, ds = closest_edge([z[next_edge[1]], x[next_edge[2]], y[next_edge[3]]],
                             r, unit_vector)

    velocity_los = sum(velocity[box_id...] .* unit_vector)

    α = α_continuum[box_id...] +
        line_extinction(λ, λ0, ΔλD[box_id...], damping_constant[box_id...],
                        α_line_constant[box_id...], velocity_los)

    τ_cum = ds * α
    r += ds * unit_vector

    # ===================================================================
    # TRAVERSE BOXES UNTIL DEPTH TARGET REACHED
    # ===================================================================
    while τ > τ_cum
        # Switch to new box
        box_id[face] += direction[face]
        next_edge[face] += direction[face]

        # Check if escaped
        if face == 1
            if box_id[1] == 0
                lost = true

                if θ < π/32
                    I0[1,box_id[2], box_id[3]] += 1
                elseif π/32 < θ < π/4
                    I0[2,box_id[2], box_id[3]] += 1
                else
                    I0[3,box_id[2], box_id[3]] += 1
                end

                break
            end
        # Handle side escapes with periodic boundary
        else
            # Left-going packets
            if box_id[face] == 0
                box_id[face] = side_dim[face-1]
                next_edge[face] = side_dim[face-1]
                r[face] = side_edge[face-1,2]
            # Right-going packets
            elseif box_id[face] == side_dim[face-1] + 1
                box_id[face] = 1
                next_edge[face] = 2
                r[face] = side_edge[face-1,1]
            end
        end

        # Check that above boundary
        if box_id[1] > boundary[box_id[2], box_id[3]]
            lost = true
            break
        end

        # Add to radiation field
        J[box_id...] += 1

        # Closest face and distance to it
        face, ds = closest_edge([z[next_edge[1]], x[next_edge[2]], y[next_edge[3]]],
                                 r, unit_vector)

        velocity_los = sum(velocity[box_id...] .* unit_vector)
        α = α_continuum[box_id...] +
            line_extinction(λ, λ0, ΔλD[box_id...], damping_constant[box_id...],
                            α_line_constant[box_id...], velocity_los)

        τ_cum += ds * α
        r += ds * unit_vector
    end

    # ===================================================================
    # CORRECT FOR OVERSHOOT IN FINAL BOX
    # ===================================================================
    if !lost
        r -= unit_vector*(τ_cum - τ)/α
    end

    return box_id, r, lost, α
end

"""
    closest_edge(next_edges::Array{<:Unitful.Length, 1},
                 r::Array{<:Unitful.Length, 1},
                 unit_vector::Array{Float64, 1})

Returns the face (1=z,2=x or 3=y) that
the packet will cross next and the distance to it.
"""
function closest_edge(next_edges::Array{<:Unitful.Length, 1},
                      r::Array{<:Unitful.Length, 1},
                      unit_vector::Array{Float64, 1})

    # Distance to next face cross in all dimensions
    distance = (next_edges .- r) ./unit_vector

    # Closest face cross
    face = argmin(distance)
    ds = distance[face]

    return face, ds
end


"""
     mcrt_continuum(atmosphere::Atmosphere,
                    radiation::Radiation,
                    λ::Array{<:Unitful.Length, 1},
                    max_scatterings::Real,
                    iteration::Int64,
                    nλ0::Int64,
                    Nλ::Int64,
                    output_path::String)

Monte Carlo radiative transfer simulation
for continuum radiation."""

function mcrt_continuum(atmosphere::Atmosphere,
                        radiation::Radiation,
                        λ::Array{<:Unitful.Length, 1},
                        max_scatterings::Real,
                        iteration::Int64,
                        output_path::String)

    # ==================================================================
    # ATMOSPHERE DATA
    # ==================================================================
    x = atmosphere.x
    y = atmosphere.y
    z = atmosphere.z

    # ===================================================================
    # RADIATION DATA
    # ===================================================================
    α = radiation.α_continuum
    ε = radiation.ε_continuum
    boundary = radiation.boundary
    packets = radiation.packets

    # ===================================================================
    # SET UP VARIABLES
    # ===================================================================
    nλ, nz, nx, ny = size(α)

    # Initialise placeholder variable
    J_λ = zeros(Float64, nz, nx, ny)
    I0_λ = zeros(Int64, 3, nx, ny)

    # ===================================================================
    # SIMULATION
    # ===================================================================
    println(@sprintf("--Starting simulation, using %d thread(s)...",
            Threads.nthreads()))

    for λi=1:nλ

        # Reset counters
        fill!(J_λ, 0.0)
        fill!(I0_λ, 0.0)
        total_destroyed = Threads.Atomic{Int64}(0)
        total_scatterings = Threads.Atomic{Int64}(0)

        # Pick out wavelength data
        packets_λ = packets[λi,:,:,:]
        #J_λ .= packets_λ
        α_λ = α[λi,:,:,:]
        boundary_λ = boundary[λi,:,:]
        ε_λ = ε[λi,:,:,:]

        println("\n--[",λi,"/",nλ, "]        ", @sprintf("λ = %.3f nm", ustrip(λ[λi])))

        # Create ProgressMeter working with threads
        p = Progress(ny); update!(p,0)
        jj = Threads.Atomic{Int}(0)
        l = Threads.SpinLock()

        # Go through all boxes
        et = @elapsed Threads.@threads for j=1:ny

            # Advance ProgressMeter
            Threads.atomic_add!(jj, 1)
            Threads.lock(l); update!(p, jj[])
            Threads.unlock(l)

            local_destroyed = 0
            local_scattered = 0

            for i=1:nx
                for k=1:boundary_λ[i,j]

                    # Packets in box
                    pcts = Int(packets_λ[k,i,j])

                    if pcts == 0
                        continue
                    end

                    # Dimensions of box
                    corner = [z[k], x[i], y[j]]
                    box_dim = [z[k+1], x[i+1], y[j+1]] .- corner

                    for pct=1:pcts
                        # Initial box
                        box_id = [k,i,j]

                        # Initial position uniformely drawn from box
                        r = corner .+ (box_dim .* rand(3))

                        # Scatter each packet until destroyed,
                        # escape or reach max_scatterings
                        for s=1:Int(max_scatterings)

                            # Scatter packet once
                            box_id, r, lost = scatter_packet_continuum(x, y, z,
                                                                       α_λ,
                                                                       boundary_λ,
                                                                       box_id, r,
                                                                       J_λ, I0_λ)

                            # Check if escaped or lost in bottom
                            if lost
                                break
                            # Check if destroyed in next particle interaction
                            elseif rand() < ε_λ[box_id...]
                                local_destroyed += 1
                                break
                            end
                            local_scattered += 1

                        end
                    end
                end
                for k=(boundary_λ[i,j]+1):nz
                    J_λ[k,i,j] = packets_λ[k,i,j]
                end
            end
            Threads.atomic_add!(total_destroyed, local_destroyed)
            Threads.atomic_add!(total_destroyed, local_scattered)
        end

        # ===================================================================
        # WRITE TO FILE
        # ===================================================================
        h5open(output_path, "r+") do file
            file["J"][iteration,λi,:,:,:] = J_λ
            file["I0"][iteration,λi,:,:,:] = I0_λ
            file["total_destroyed"][iteration,λi] = total_destroyed.value
            file["total_scatterings"][iteration,λi] = total_scatterings.value
            file["total_destroyed"][iteration,λi] = total_destroyed.value
            file["time"][iteration,λi] = et
        end
    end
end

"""
    scatter_packet_continuum(x::Array{<:Unitful.Length, 1},
                             y::Array{<:Unitful.Length, 1},
                             z::Array{<:Unitful.Length, 1},
                             α::Array{<:PerLength, 3},
                             boundary::Array{Int32, 2},
                             box_id::Array{Int64,1},
                             r::Array{<:Unitful.Length, 1},
                             J::Array{Int32, 3},
                             I0::Array{Int32, 3})

Traverse boxes until optical depth target reached
for scattering event and return new position.
"""
function scatter_packet_continuum(x::Array{<:Unitful.Length, 1},
                                  y::Array{<:Unitful.Length, 1},
                                  z::Array{<:Unitful.Length, 1},
                                  α::Array{<:PerLength, 3},
                                  boundary::Array{Int32, 2},
                                  box_id::Array{Int64,1},
                                  r::Array{<:Unitful.Length, 1},
                                  J::Array{Float64, 3},
                                  I0::Array{Int64, 3})

    # Keep track of status
    lost = false

    # Useful quantities
    side_dim = SVector(size(boundary))
    side_edge = SA[x[1] x[end]
                   y[1] y[end]]

    # ===================================================================
    # DRAW DEPTH AND DIRECTION
    # ===================================================================

    # Draw scattering depth and direction
    τ = -log(rand())
    ϕ = 2π * rand()
    θ =  π * rand()

    # Find direction
    unit_vector = [cos(θ), sin(θ)*cos(ϕ), sin(θ)*sin(ϕ)]
    direction = Int.(sign.(unit_vector))
    direction[1] = -direction[1] # because height array up->down

    # ===================================================================
    # MOVE PACKET TO FIRST BOX INTERSECTION
    # ===================================================================

    # Next face cross in all dimensions
    next_edge = (direction .> 0) .+ box_id

    # Closest face and distance to it
    face, ds = closest_edge([z[next_edge[1]], x[next_edge[2]], y[next_edge[3]]],
                             r, unit_vector)

    τ_cum = ds * α[box_id...]
    r += ds * unit_vector

    # ===================================================================
    # TRAVERSE BOXES UNTIL DEPTH TARGET REACHED
    # ===================================================================
    # If packet has next interaction in same box
    """if τ < τ_cum
        J[box_id...] += 1
    end"""

    while τ > τ_cum

        # Switch to new box
        box_id[face] += direction[face]
        next_edge[face] += direction[face]

        # Check if escaped
        if face == 1
            if box_id[1] == 0
                lost = true

                if θ < π/32
                    I0[1,box_id[2], box_id[3]] += 1
                elseif π/32 < θ < π/4
                    I0[2,box_id[2], box_id[3]] += 1
                else
                    I0[3,box_id[2], box_id[3]] += 1
                end

                break
            end

        # Handle side escapes with periodic boundary
        else
            # Left-going packets
            if box_id[face] == 0
                box_id[face] = side_dim[face-1]
                next_edge[face] = side_dim[face-1]
                r[face] = side_edge[face-1,2]
            # Right-going packets
            elseif box_id[face] == side_dim[face-1] + 1
                box_id[face] = 1
                next_edge[face] = 2
                r[face] = side_edge[face-1,1]
            end
        end

        # Check that above boundary
        if box_id[1] > boundary[box_id[2], box_id[3]]
            lost = true
            break
        end

        # Add to radiation field
        J[box_id...] += 1

        # Closest face and distance to it
        face, ds = closest_edge([z[next_edge[1]], x[next_edge[2]], y[next_edge[3]]],
                                 r, unit_vector)

        τ_cum += ds * α[box_id...]
        r += ds * unit_vector
    end

    # ===================================================================
    # CORRECT FOR OVERSHOOT IN FINAL BOX
    # ===================================================================
    if !lost
        r -= unit_vector*(τ_cum - τ)/α[box_id...]
    end

    return box_id, r, lost
end
