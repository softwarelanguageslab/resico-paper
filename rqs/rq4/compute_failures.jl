
function read_failures(dataset_name::String, folder_name::String, mapping::Dict{Int64, String}, needs_mapping::Bool)
    lines_data = readlines("/Users/kmilo/Dev/PhD/RESICO_new/rq2/results/$folder_name/1/$dataset_name.txt")

    failures_dict = Dict{String, Vector{String}}()
    for line in lines_data
        line_divided = split(line, ",")
        
        true_value = line_divided[1]
        predicted_value = line_divided[2]

        if needs_mapping
            true_value = mapping[parse(Int64, true_value)]
            predicted_value = mapping[parse(Int64, predicted_value)]
        end

        if predicted_value != true_value
            current_vector = Vector{String}()
            if haskey(failures_dict, true_value)
                current_vector = failures_dict[true_value]
            end
            push!(current_vector, predicted_value)
            failures_dict[true_value] = current_vector
        end
    end

    return failures_dict
end

function main()
    mapping_data = readlines("/Users/kmilo/Dev/PhD/RESICO_new/rq1/results/models/resico/mapping.txt")
    unique_mapping = unique(mapping_data)

    fqns_dict = Dict{Int64, String}()
    for line in unique_mapping
        line_divided = split(line, ",")
        key_converted = parse(Int64, line_divided[1])
        fqns_dict[key_converted] = line_divided[2]
    end

    FOLDER_NAME = "resico"
    FAILURES_PATH = "/Users/kmilo/Dev/PhD/RESICO_new/rq4/failures/$FOLDER_NAME"
    external_datasets = ["COSTER-SO", "StatType-SO", "RESICO-SO"]

    for external_dataset in external_datasets
        println("Analysing predictions for $external_dataset ...")
        failures = read_failures(external_dataset, FOLDER_NAME, fqns_dict, true)

        # Sorting the failures by their frequency
        sorted_failures = sort(collect(failures), by=x->length(x[2]), rev=true)
        file_failures = open("$FAILURES_PATH/$external_dataset.txt", "w")
        file_failures_counting = open("$FAILURES_PATH/$(external_dataset)_counting.txt", "w")

        println("Number of misclassified FQNs: $(length(failures))")
        quantity_failures = 0
        predicted_ambiguity = 0
        ambiguity_misclassifications = Vector{String}()

        for (failure, vector_failure) in sorted_failures
            line_failure = failure * "->" * join(vector_failure, ",") * "\n"
            quantity_failure = length(vector_failure)
            quantity_failures += quantity_failure
            line_failure_counting = failure * "->" * string(quantity_failure) * "\n"

            simple_name_failure = last(split(failure, "."))

            for misprediction in vector_failure
                simple_name_misprediction = last(split(misprediction, "."))

                if simple_name_failure == simple_name_misprediction
                    predicted_ambiguity += 1
                    push!(ambiguity_misclassifications, "$failure misclassified as $misprediction")
                end
            end

            write(file_failures, line_failure)
            write(file_failures_counting, line_failure_counting)
        end

        println("Total number of misclassifications: $quantity_failures")
        println("Total number of mispredicted ambiguities: $predicted_ambiguity")

        for ambiguity in ambiguity_misclassifications
            println("\t$ambiguity")
        end

        println("Percentage of misclassifications that are ambiguities: $(round((predicted_ambiguity / quantity_failures) * 100.0, digits=2))%")
        println()

        close(file_failures)
        close(file_failures_counting)
    end
end

main()
