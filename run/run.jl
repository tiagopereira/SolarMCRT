include("../src/mcrt.jl")
include("../src/populations.jl")

function run()
    println("\n", "="^91, "\n", " "^34,
            "SOLAR ATMOSPHERE MCRT",
            "\n", "="^91, "\n")

    input = YAML.load_file("input.yaml")

    # =============================================================================
    # LOAD ATMOSPHERE DATA
    # =============================================================================
    print("--Loading atmosphere data..................")
    atmosphere_parameters = collect_atmosphere(input)  ## Needs work, probably feed input to handle inside
    atmosphere = Atmosphere(atmosphere_parameters...)
    atmosphere_size = size(atmosphere.temperature)
    println("Atmosphere loaded with dimensions ", atmosphere_size, ".")


    # =============================================================================
    # READ CONFIG FILE
    # =============================================================================
    output_path = get_output_path(input)
    max_scatterings = input["max_scatterings"]
    boundary_config = [input["depth_criterion"], input["depth_exponent"]]
    packet_config = [input["target_packets"], input["packet_exponent"]]

    if input["background_mode"]
        # =============================================================================
        # LOAD WAVELENGTH
        # =============================================================================
        print("--Loading wavelength.......................")
        λ = [input["background_wavelength"] * u"nm"]
        nλ = length(λ)
        println("Wavelength λ = ", λ[1], " loaded.")

        # =============================================================================
        # LOAD RADIATION DATA
        # =============================================================================
        print("--Loading radiation data...................")
        radiation_parameters = collect_background_radiation(atmosphere, λ, boundary_config, packet_config)
        radiation = Radiation(radiation_parameters...)
        println(@sprintf("Radiation loaded with %.2e packets.", packet_config[1]))

        # =============================================================================
        # CREATE OUTPUT FILE
        # =============================================================================
        print("--Initialise output file...................")
        create_output_file(output_path, nλ, atmosphere_size)
        write_to_file(λ, output_path)
        write_to_file(radiation, output_path)
        println(@sprintf("%.1f GBs of data initialised.", how_much_data(nλ, atmosphere_size)))

        # =============================================================================
        # SIMULATION
        # =============================================================================
        mcrt_continuum(atmosphere, radiation, λ, max_scatterings, 1, output_path)

        # =============================================================================
        # END OF TEST MODE
        # =============================================================================

    else
        # =============================================================================
        # READ CONFIG FILE
        # =============================================================================
        max_iterations = input["max_iterations"]
        initial_population_distribution = input["population_distribution"]
        write_rates = input["write_rates"]

        # =============================================================================
        # LOAD ATOM
        # =============================================================================
        print("--Loading atom.............................")
        atom_parameters = collect_atom_data(atmosphere, input)
        atom = Atom(atom_parameters...)
        println("Atom loaded with ", atom.nλ , " wavelengths.")

        n_levels = atom.n_levels
        n_lines = atom.n_lines
        n_transitions = n_levels + n_lines

        # =============================================================================
        # LOAD LINES
        # =============================================================================
        print("--Loading line.............................")
        lines = []
        for l=1:n_levels-1
            for u=(l+1):n_levels
                line =  Line(collect_line_data(atmosphere, atom, u, l)...)
                append!(lines, [line])
            end
        end
        println(n_lines, " line(s) loaded.")

        # =============================================================================
        # LOAD INITIAL POPULATIONS
        # =============================================================================
        print("--Loading initial populations..............")
        populations = collect_initial_populations(atmosphere, atom, lines, initial_population_distribution)
        println("Initial ", initial_population_distribution, "-populations loaded.")

        # =============================================================================
        # CALCULATE INITIAL TRANSITION RATES
        # =============================================================================
        print("--Loading initial transition rates.........")
        Bλ = blackbody_lambda(atom.λ, atmosphere.temperature)
        rate_parameters = calculate_transition_rates(atmosphere, atom, lines, Bλ)
        rates = TransitionRates(rate_parameters...)
        println("Initial transition rates loaded.")

        # =============================================================================
        # CREATE OUTPUT FILE
        # =============================================================================
        print("--Initialise output file...................")
        create_output_file(output_path, max_iterations, atom.nλ, n_levels, atmosphere_size, write_rates)
        write_to_file(atom.λ, atom.iλbf, atom.iλbb, output_path)
        write_to_file(populations, 0, output_path)
        if write_rates; write_to_file(rates, 0, output_path); end
        println(@sprintf("%.1f GBs of data initialised.", how_much_data(atom.nλ, atmosphere_size, max_iterations, write_rates)))

        # =============================================================================
        # RUN MCRT UNTIL POPULATIONS CONVERGE
        # =============================================================================

        for n=1:max_iterations
            println("\n  ITERATION ", n, "\n", "="^91, "\n", "="^91)

            print("--Loading radiation data...................")
            lineRadiations = []

            for l=1:n_levels-1
                for u=(l+1):n_levels
                    line_number = sum((n_levels-l+1):(n_levels-1)) + (u - l)
                    lineRadiation = LineRadiation(collect_line_radiation_data(lines[line_number], rates, populations)...)
                    append!(lineRadiations, [lineRadiation])
                end
            end

            radiation_parameters = collect_radiation(atmosphere, atom, rates, lines, lineRadiations,
                                                     populations, boundary_config, packet_config)

            radiation = Radiation(radiation_parameters...)
            write_to_file(radiation, n, output_path)
            println(@sprintf("Radiation loaded with %.2e packets per λ.", packet_config[1]))


            mcrt(atmosphere, radiation, atom,
                 lines, lineRadiations,
                 max_scatterings, n, output_path)

            # =============================================================================
            # CALCULATE NEW TRANSITION RATES
            # =============================================================================
            print("\n--Update transition rates..................")
            Jλ = get_Jλ(output_path, n, atom.λ)
            rate_parameters = calculate_transition_rates(atmosphere, atom, lines, Jλ)
            rates = TransitionRates(rate_parameters...)
            if write_rates; write_to_file(rates, n, output_path); end
            println("Transition rates updated.")

            # =============================================================================
            # CALCULATE NEW POPULATIONS
            # =============================================================================
            print("--Update populations.......................")
            new_populations = get_revised_populations(rates, atom.density)
            write_to_file(new_populations, n, output_path)
            println("Populations updated.")

            # =============================================================================
            # CHECK FOR UNVALID VARIABLES
            # =============================================================================
            @test all( Inf .> ustrip.(new_populations) .>= 0.0 )
            @test all( Inf .> ustrip.(rates.R) .>= 0.0 )
            @test all( Inf .> ustrip.(rates.C) .>= 0.0 )

            # =============================================================================
            # CHECK POPULATION CONVERGENCE
            # =============================================================================
            converged, error = check_population_convergence(populations, new_populations)
            populations = copy(new_populations)

            if converged
                println(@sprintf("--Convergence at iteration n = %d. Error = %.1e.\n", n, error))
                cut_output_file(output_path, n, write_rates)
                break
            else
                println(@sprintf("\n--No convergence. Mean error = %.1e.\n", error))
            end

            # =============================================================================
            # END OF ITERATION
            # =============================================================================
        end

        # =============================================================================
        # END OF ATOM MODE
        # =============================================================================
    end

end

run()
