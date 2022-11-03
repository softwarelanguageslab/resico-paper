using ArgParse

function parse_commandline()
    settings = ArgParseSettings()

    @add_arg_table settings begin
        "--file", "-f"
            help = "name of the file from which stats are going to be extracted"
            required = true
    end

    return parse_args(settings)
end

function main()
    parse_args = parse_commandline()

    dataset_name = parse_args["file"]
    println("Getting the stats for $dataset_name ...")

    lines = readlines(dataset_name)
    true_classes = map(lines) do line
        line_divided = split(line, ",")
        return line_divided[1]
    end

    unique_classes = unique(true_classes)
    counter_dict = Dict([(class_element, count(x -> x == class_element, true_classes)) for class_element in unique_classes])

    ordered_dict = sort(collect(counter_dict), by = x -> x[2], rev = true)

    internal_fqn = 0
    external_fqn = 0

    unique_internal = 0
    unique_external = 0

    startswith_javalang = startswith("java.lang.")
    startswith_javaio = startswith("java.io.")

    for (element, value) in ordered_dict
        if startswith_javalang(element) || startswith_javaio(element)
            class_element = last(split(element, "."))
            if isuppercase(class_element[1])
                internal_fqn += value
                unique_internal += 1
            else
                external_fqn += value
                unique_external += 1
            end
        else
            external_fqn += value
            unique_external += 1
        end
    end

    printstyled("Number of Internal FQNs (i.e., no need for prediction): $internal_fqn ($(internal_fqn / length(lines) * 100)}%)\n", color = :green)
    printstyled("Number of Unique Internal FQNs: $unique_internal\n", color = :green)

    printstyled("Number of External FQNs: $external_fqn ($(external_fqn / length(lines) * 100)}%\n", color = :green)
    printstyled("Number of Unique External FQNs: $unique_external\n", color = :green)
end

main()
