using DelimitedFiles
using BenchmarkTools
using ProgressMeter
using Transparency
using StaticArrays
using Unitful
using Random
using Printf
using Test
using HDF5
import PeriodicTable
import YAML


import PhysicalConstants.CODATA2018: c_0, h, k_B, m_u, m_e, R_∞, ε_0, e
const E_∞ = R_∞ * c_0 * h
const hc = h * c_0

@derived_dimension NumberDensity Unitful.𝐋^-3
@derived_dimension PerLength Unitful.𝐋^-1
@derived_dimension PerArea Unitful.𝐋^-2
@derived_dimension PerTime Unitful.𝐓^-1
@derived_dimension UnitsIntensity_λ Unitful.𝐋^-1 * Unitful.𝐌 * Unitful.𝐓^-3

"""
    background_mode()

Get the test mode status.
"""
function background_mode()
    return input["background_mode"]
end


"""
    get_output_path()

Get the path
 to the output file.
"""
function get_output_path(input::Dict)

	path = "../out/output_"

	crit, depth_exp = [input["depth_criterion"], input["depth_exponent"]]
	target_packets, pct_exp = [input["target_packets"], input["packet_exponent"]]

	if crit != false
		path *= string(crit)*"_"*string(depth_exp)
	end

    if input["background_mode"]
        path *= "_" * string(input["background_wavelength"]) * "nm_" * string(target_packets) * "pcs.h5"
    else
		pop_distrib = input["population_distribution"]
        nλ_bb = input["nλ_bb"]
	    nλ_bf = input["nλ_bf"]

		for i=1:length(nλ_bf)
			path *= "_"*string(nλ_bf[i])
		end

		for i=1:length(nλ_bb)
			path *= "_"*string(nλ_bb[i])
		end

	path *= "_"*string(target_packets)*"_"*string(pct_exp)

        if pop_distrib == "LTE"
            path *= "_LTE.h5"
        elseif pop_distrib == "zero_radiation"
            path *= "_ZR.h5"
        end
    end

    return path
end


# =============================================================================
# RADIATION
# =============================================================================
"""
    get_Jλ(output_path::String, iteration::Int64, intensity_per_packet::Array{<:UnitsIntensity_λ,1})

Get the radiation field from the output file.
"""
function get_Jλ(output_path::String, iteration::Int64, λ)

    file = h5open(output_path, "r")
    J = read(file, "J")[iteration,:,:,:,:]
	packets_to_intensity = read(file, "packets_to_intensity")[iteration,:,:,:,:] .*u"kW / m^2 / sr / nm"
    close(file)

    return J .* packets_to_intensity
end


# =============================================================================
# OUTPUT FILE
# =============================================================================

"""
    create_output_file(output_path::String, max_iterations::Int64, nλ::Int64, atmosphere_size::Tuple, write_rates::Bool)
Initialise all output variables for the full atom mode.
"""
function create_output_file(output_path::String, max_iterations::Int64, nλ::Int64, n_levels, atmosphere_size::Tuple, write_rates::Bool)

    nz, nx, ny = atmosphere_size

    h5open(output_path, "w") do file
        write(file, "J", Array{Float64,5}(undef, max_iterations, nλ, nz, nx,ny))
		write(file, "I0", Array{Int64,5}(undef, max_iterations, nλ, 3, nx,ny))
        write(file, "total_destroyed", Array{Int64,2}(undef, max_iterations, nλ))
        write(file, "total_scatterings", Array{Int64,2}(undef,max_iterations, nλ))
        write(file, "time", Array{Float64,2}(undef,max_iterations, nλ))

        write(file, "packets", Array{Float64,5}(undef, max_iterations, nλ, nz, nx, ny))
        write(file, "boundary", Array{Int32,4}(undef,max_iterations, nλ, nx, ny))
        write(file, "packets_to_intensity", Array{Float64,5}(undef,max_iterations, nλ,nz, nx, ny))

        write(file, "populations", Array{Float64,5}(undef, max_iterations+1, nz, nx, ny, n_levels+1))
		write(file, "error", Array{Float64,1}(undef, max_iterations))

        if write_rates
            write(file, "R", Array{Float64,6}(undef, max_iterations+1, n_levels+1, n_levels+1, nz,nx,ny))
            write(file, "C", Array{Float64,6}(undef, max_iterations+1, n_levels+1, n_levels+1, nz,nx,ny))
        end
    end
end

"""
    create_output_file(output_path::String, nλ::Int64, atmosphere_size::Tuple)

Initialise all output variables for the test mode.
"""
function create_output_file(output_path::String, nλ::Int64, atmosphere_size::Tuple)

    nz, nx, ny = atmosphere_size

    h5open(output_path, "w") do file
        write(file, "J", Array{Float64,5}(undef, 1, nλ, nz, nx,ny))
        write(file, "I0", Array{Int64,5}(undef, 1, nλ, 3, nx,ny))
		write(file, "total_destroyed", Array{Int64,2}(undef,1, nλ))
        write(file, "total_scatterings", Array{Int64,2}(undef,1, nλ))
        write(file, "time", Array{Float64,2}(undef,1, nλ))

        write(file, "packets", Array{Float64,5}(undef,1, nλ, nz, nx, ny))
        write(file, "boundary", Array{Int32,4}(undef, 1,nλ, nx, ny))
        write(file, "packets_to_intensity", Array{Float64,5}(undef,1, nλ, nz, nx, ny))
    end
end

"""
    cut_output_file(output_path::String, final_iteration::Int64, write_rates::Bool)

Cut output data at a given iteration.
"""
function cut_output_file(output_path::String, final_iteration::Int64, write_rates::Bool)
    h5open(output_path, "r+") do file
        # Slice
        J_new = read(file, "J")[1:final_iteration,:,:,:,:]
		I0_new = read(file, "I0")[1:final_iteration,:,:,:,:]
        total_destroyed_new = read(file, "total_destroyed")[1:final_iteration,:]
        total_scatterings_new = read(file, "total_scatterings")[1:final_iteration,:]
        time_new = read(file, "time")[1:final_iteration,:]
        packets_new = read(file, "packets")[1:final_iteration,:,:,:,:]
        boundary_new = read(file, "boundary")[1:final_iteration,:,:,:]
        intensity_per_packet_new = read(file, "packets_to_intensity")[1:final_iteration,:,:,:,:]
        populations_new = read(file, "populations")[1:final_iteration+1,:,:,:,:]

        # Delete
        delete_object(file, "J")
		delete_object(file, "I0")
        delete_object(file, "total_destroyed")
        delete_object(file, "total_scatterings")
        delete_object(file, "time")
        delete_object(file, "packets")
        delete_object(file, "boundary")
        delete_object(file, "packets_to_intensity")
        delete_object(file, "populations")

        # Write
        write(file, "J", J_new)
		write(file, "I0", I0_new)
        write(file, "total_destroyed", total_destroyed_new)
        write(file, "total_scatterings", total_scatterings_new)
        write(file, "time", time_new)
        write(file, "packets", packets_new)
        write(file, "boundary", boundary_new)
        write(file, "packets_to_intensity", intensity_per_packet_new)
        write(file, "populations", populations_new)

        if write_rates
            R_new = read(file, "R")[1:final_iteration,:,:,:,:,:]
			C_new = read(file, "C")[1:final_iteration,:,:,:,:,:]

            delete_object(file, "R")
            delete_object(file, "C")

            write(file, "R", R_new)
            write(file, "C", C_new)
        end
    end
end

"""
    how_much_data(max_iterations::Int64, nλ::Int64, atmosphere_size::Tuple, write_rates::Bool)

Returns the maximum amount of GBs written to file if the
simulation runs for max_iterations.
"""
function how_much_data(nλ::Int64, atmosphere_size::Tuple, max_iterations::Int64, write_rates::Bool)

    nz, nx, ny = atmosphere_size
    boxes = nz*nx*ny
    slice = nx*ny

    λ_data = 8*nλ + 8*2

    # Iteration data
    J_data   = 4boxes*nλ
    sim_data = 4nλ + 2 * 8nλ
    rad_data = 4boxes*nλ + 4slice*nλ + 8nλ
    pop_data = 8boxes*3

    # Rates
    rate_data = 8*12boxes #fix

    max_data = ( λ_data +
               ( J_data + sim_data + rad_data) * max_iterations +
                                      pop_data * (max_iterations + 1) ) / 1e9

    if write_rates
        max_data += rate_data * (max_iterations+1)/1e9
    end

    return max_data
end

"""
    how_much_data(max_iterations::Int64, nλ::Int64, atmosphere_size::Tuple)

Returns the maximum amount of GBs written
to file for the background mode.
"""
function how_much_data(nλ::Int64, atmosphere_size::Tuple)

    nz, nx, ny = atmosphere_size
    boxes = nz*nx*ny
    slice = nx*ny

    λ = 8*nλ

    # Iteration data
    J = 8boxes*nλ
	B = 8boxes*nλ
    sim_data = 4nλ + 2 * 8nλ
    rad_data = 4boxes*nλ + 4slice*nλ + 8nλ

    max_data = ( λ +  J + sim_data + rad_data ) / 1e9

    return max_data
end
