
function main()
    mapping_data = readlines("/Users/kmilo/Dev/PhD/RESICO_new/rq1/results/models/resico/mapping.txt")
    println("Number of FQNs in the data: $(length(mapping_data))")

    unique_mapping = unique(mapping_data)
    println("Unique FQNs in the data: $(length(unique_mapping))")

    fqns = map(unique_mapping) do line
        line_divided = split(line, ",")
        return line_divided[2]
    end

    simple_names = map(fqns) do fqn
        fqn_divided = split(fqn, ".")
        return last(fqn_divided)
    end

    unique_simple_names = unique(simple_names)

    println("Unique Simple Names from the unique FQNs: $(length(unique_simple_names))")

    println("Extracting the mapping of simple names in the data ...")
    names_dict = Dict{String, Vector{String}}()
    for (index, simple_name) in enumerate(unique_simple_names)
        if index % 1_000 == 0
            println(index)
        end

        fqns_simple_name = Vector()
        for fqn in fqns
            fqn_divided = split(fqn, ".")
            simple_name_temp = last(fqn_divided)

            if simple_name == simple_name_temp
                push!(fqns_simple_name, fqn)
            end
        end

        names_dict[simple_name] = fqns_simple_name
    end

    println("Done!")

    println("Sorting by highest frequency and writing to files ...")
    sorted_names = sort(collect(names_dict), by=x->length(x[2]), rev=true)

    file_simple_names = open("simple_names_mapping.txt", "w")
    file_simple_names_counting = open("simple_names_counting.txt", "w")

    for (simple_name, vector_fqn) in sorted_names
        line_names = simple_name * "->" * join(vector_fqn, ",") * "\n"
        line_names_counting = simple_name * "->" * string(length(vector_fqn)) * "\n"

        write(file_simple_names, line_names)
        write(file_simple_names_counting, line_names_counting)
    end

    close(file_simple_names)
    close(file_simple_names_counting)
    println("Done!")
end

main()
