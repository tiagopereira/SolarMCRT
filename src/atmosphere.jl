include("io.jl")

struct Atmosphere
    z::Array{<:Unitful.Length, 1}                       # (nz + 1)
    x::Array{<:Unitful.Length, 1}                       # (nx + 1)
    y::Array{<:Unitful.Length, 1}                       # (ny + 1)
    velocity::Array{Array{<:Unitful.Velocity, 1}, 3}    # (nx, ny, nz)
    velocity_z::Array{<:Unitful.Velocity, 3}            # (nx, ny, nz)
    temperature::Array{<:Unitful.Temperature, 3}        # (nx, ny, nz)
    electron_density::Array{<:NumberDensity, 3}         # (nx, ny, nz)
    hydrogen_populations::Array{<:NumberDensity, 4}     # (nx, ny, nz, 3)
end


"""
    collect_atmosphere_data()

Reads and slices atmosphere parameters according to
keyword.input file. Returns atmosphere dimensions, velocity,
temperature, electron_density and hydrogen populations.
"""
function collect_atmosphere(input::Dict)

    # ===========================================================
    # READ ATMOSPHERE FILE
    # ===========================================================
    h5open(input["atmosphere_path"], "r") do atmos
        x = read(atmos, "x")u"m"
        y = read(atmos, "y")u"m"
        z = read(atmos, "z")u"m"

        velocity_x = read(atmos, "velocity_x")u"m/s"
        velocity_y = read(atmos, "velocity_y")u"m/s"
        velocity_z = read(atmos, "velocity_z")u"m/s"

        temperature = read(atmos, "temperature")u"K"
        electron_density = read(atmos, "electron_density")u"m^-3"
        hydrogen_populations = read(atmos, "hydrogen_populations")u"m^-3"
    end

    # ===========================================================
    # GET SNAPSHOT (remove)
    # ===========================================================

    if length(size(z)) == 2
        z = z[:,1]
        velocity_x = velocity_x[:,:,:,1]
        velocity_y = velocity_y[:,:,:,1]
        velocity_z = velocity_z[:,:,:,1]
        temperature = temperature[:,:,:,1]
        electron_density = electron_density[:,:,:,1]
        hydrogen_populations = hydrogen_populations[:,:,:,:,1]
    end

    # ===========================================================
    # MAKE SURE Δx, Δy > 0 and Δz < 0 (remove)
    # ===========================================================
    if z[1] < z[end]
        z = reverse(z)
        velocity_z = velocity_z[end:-1:1,:,:]
        velocity_x = velocity_x[end:-1:1,:,:]
        velocity_y = velocity_y[end:-1:1,:,:]
        temperature = temperature[end:-1:1,:,:]
        electron_density = electron_density[end:-1:1,:,:]
        hydrogen_populations = hydrogen_populations[end:-1:1,:,:,:]
    end

    if x[1] > x[end]
        x = reverse(x)
        velocity_z = velocity_z[:,end:-1:1,:]
        velocity_x = velocity_x[:,end:-1:1,:]
        velocity_y = velocity_y[:,end:-1:1,:]
        temperature = temperature[:,end:-1:1,:]
        electron_density = electron_density[:,end:-1:1,:]
        hydrogen_populations = hydrogen_populations[:,end:-1:1,:,:]
    end


    if y[1] > y[end]
        y = reverse(y)
        velocity_z = velocity_z[:,:,end:-1:1]
        velocity_x = velocity_x[:,:,end:-1:1]
        velocity_y = velocity_y[:,:,end:-1:1]
        temperature = temperature[:,:,end:-1:1]
        electron_density = electron_density[:,:,end:-1:1]
        hydrogen_populations = hydrogen_populations[:,:,end:-1:1,:]
    end

    # ===========================================================
    # CUT AND SLICE ATMOSPHERE BY INDEX
    # ===========================================================

    # original dimensions of data
    nz, nx, ny = size(temperature)

    ze, xe, ye = input["stop"]
    zs, xs, ys = input["start"]
    dz, dx, dy = input["step"]

    # Cut z-direction from below
    if isa(ze, Number) && ze < nz
        nz = ze
        z = z[1:nz+1]
        velocity_x = velocity_x[1:nz,:,:]
        velocity_y = velocity_y[1:nz,:,:]
        velocity_z = velocity_z[1:nz,:,:]
        temperature = temperature[1:nz,:,:]
        electron_density = electron_density[1:nz,:,:]
        hydrogen_populations = hydrogen_populations[1:nz,:,:,:]
    end

    # Cut  z-direction from up top
    if zs > 1
        nz = zs
        z = z[nz:end]
        velocity_x = velocity_x[nz:end,:,:]
        velocity_y = velocity_y[nz:end,:,:]
        velocity_z = velocity_z[nz:end,:,:]
        temperature = temperature[nz:end,:,:]
        electron_density = electron_density[nz:end,:,:]
        hydrogen_populations = hydrogen_populations[nz:end,:,:,:]
    end

    # Cut x-direction from right
    if isa(xe, Number) && xe < nx
        nx = xe
        x = x[1:nx+1]
        velocity_x = velocity_x[:,1:nx,:]
        velocity_y = velocity_y[:,1:nx,:]
        velocity_z = velocity_z[:,1:nx,:]
        temperature = temperature[:,1:nx,:]
        electron_density = electron_density[:,1:nx,:]
        hydrogen_populations = hydrogen_populations[:,1:nx,:,:]
    end

    # Cut x-direction from the left
    if xs > 1
        nx = xs
        x = x[nx:end]
        velocity_x = velocity_x[:,nx:end,:]
        velocity_y = velocity_y[:,nx:end,:]
        velocity_z = velocity_z[:,nx:end,:]
        temperature = temperature[:,nx:end,:]
        electron_density = electron_density[:,nx:end,:]
        hydrogen_populations = hydrogen_populations[:,nx:end,:,:]
    end

    # Cut y-direction from right
    if isa(ye, Number) && ye < ny
        ny = ye
        y = y[1:ny+1]
        velocity_x = velocity_x[:,:,1:ny]
        velocity_y = velocity_y[:,:,1:ny]
        velocity_z = velocity_z[:,:,1:ny]
        temperature = temperature[:,:,1:ny]
        electron_density = electron_density[:,:,1:ny]
        hydrogen_populations = hydrogen_populations[:,:,1:ny,:]
    end

    # Cut y-direction from the left
    if ys > 1
        ny = ys
        y = y[ny:end]
        velocity_x = velocity_x[:,:,ny:end]
        velocity_y = velocity_y[:,:,ny:end]
        velocity_z = velocity_z[:,:,ny:end]
        temperature = temperature[:,:,ny:end]
        electron_density = electron_density[:,:,ny:end]
        hydrogen_populations = hydrogen_populations[:,:,ny:end,:]
    end

    # Only keep every dz-th box in z-direction
    if dz > 1
        z = z[1:dz:end]
        velocity_x = velocity_x[1:dz:end,:,:]
        velocity_y = velocity_y[1:dz:end,:,:]
        velocity_z = velocity_z[1:dz:end,:,:]
        temperature = temperature[1:dz:end,:,:]
        electron_density = electron_density[1:dz:end,:,:]
        hydrogen_populations = hydrogen_populations[1:dz:end,:,:,:]
    end

    # Only keep every dx-th box in x-direction
    if dx > 1
        x = x[1:dx:end]
        velocity_x = velocity_x[:,1:dx:end,:]
        velocity_y = velocity_y[:,1:dx:end,:]
        velocity_z = velocity_z[:,1:dx:end,:]
        temperature = temperature[:,1:dx:end,:]
        electron_density = electron_density[:,1:dx:end,:]
        hydrogen_populations = hydrogen_populations[:,1:dx:end,:,:]
    end

    # Only keep every dy-th box in y-direction
    if dy > 1
        y = y[1:dy:end]
        velocity_x = velocity_x[:,:,1:dy:end]
        velocity_y = velocity_y[:,:,1:dy:end]
        velocity_z = velocity_z[:,:,1:dy:end]
        temperature = temperature[:,:,1:dy:end]
        electron_density = electron_density[:,:,1:dy:end]
        hydrogen_populations = hydrogen_populations[:,:,1:dy:end,:]
    end

    # new size of data
    nz, nx, ny = size(temperature)

    # ===========================================================
    # ADD BOX END POINTS IN CASE LOST AFTER CUTTING
    # ===========================================================

    if length(z) == nz
        z = push!(z, 2*z[end] - z[end-1])
    end
    if length(x) == nx
        x = push!(x, 2*x[end] - x[end-1])
    end
    if length(y) == ny
        y = push!(y, 2*y[end] - y[end-1])
    end

    # ===========================================================
    # COLLECT VELOCITY IN ONE VARIABLE
    # ===========================================================

    velocity = Array{Array{<:Unitful.Velocity, 1}, 3}(undef,nz,nx,ny)

    for k=1:nz
        for i=1:nx
            for j=1:ny
                velocity[k,i,j] = [velocity_z[k,i,j], velocity_x[k,i,j], velocity_y[k,i,j]]
            end
        end
    end

    # ==================================================================
    # CHECK FOR UNVALID VALUES
    # =================================================================
    dz = z[2:end] .- z[1:end-1]
    dx = x[2:end] .- x[1:end-1]
    dy = y[2:end] .- y[1:end-1]
    @test all( ustrip.(dz) .<= 0.0 )
    @test all( ustrip.(dx) .>= 0.0 )
    @test all( ustrip.(dy) .>= 0.0 )
    @test all( Inf .> ustrip.(temperature) .>= 0.0 )
    @test all( Inf .> ustrip.(electron_density) .>= 0.0 )
    @test all( Inf .> ustrip.(hydrogen_populations) .>= 0.0 )

    return z, x, y, velocity, velocity_z, temperature, electron_density, hydrogen_populations
end


function box_volume(z, x, y)

    nz = length(z) - 1
    nx = length(x) - 1
    ny = length(y) - 1

    volume = Array{Unitful.Volume, 3}(undef, nz, nx, ny)

    for k = 1:nz
        for i=1:nx
            for j=1:ny
                volume[k,i,j] = (z[k] - z[k+1]) * (x[i+1] - x[i]) * (y[i+1] - y[i])
            end
        end
    end

    return volume
end
