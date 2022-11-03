using DataFrames, CSV, JLD2, Word2Vec

function obtain_vector(model_api::WordVectors, model_context::WordVectors, api_word::AbstractString, context::AbstractString, only_api::Bool=false, dimension::Int=20)
    transformed_api = replace(api_word, "|" => ".")
    transformed_context = replace(context, "|" => " ")

    vector_api = zeros(dimension)
    vector_sum_context = zeros(dimension)

    try
        vector_api = get_vector(model_api, transformed_api) 
    catch
    end

    if only_api
        return vector_api
    end

    context_divided = split(transformed_context, " ")

    for context_word in context_divided
        vector_context_word = zeros(dimension)

        try
            vector_context_word = get_vector(model_context, context_word)
        catch
        end

        vector_sum_context += vector_context_word
    end

    vector_context = vector_sum_context / length(context_divided)
    vector_averaged = (vector_api + vector_context) / 2

    return vector_averaged
end

function main()
    DATASET = "COSTER-SO"
    DATA_FOLDER = "datasets-resico"

    DATASET_TRANS = "$DATASET-T.csv"
    DATA_FOLDER_TRANS = "datasets-resico-trans"
    
    println("Loading CSV file ...")
    data = DataFrame(CSV.File("$DATA_FOLDER/$DATASET.csv"))
    println("Done!")

    println("Loading the word2vec models ...")
    api_model = wordvectors("APIsW2V.model")
    context_model = wordvectors("ContextsW2V.model")
    println("Done!")

    println("Loading the mapping of FQNs ...")
    mapping_data = unique(readlines("mapping.txt"))
    mapping = map(mapping_data) do line 
       line_divided = split(line, ",")
       line_divided[2] => parse(Int64, line_divided[1])
    end
    mapping = Dict(mapping)
    println("Done!")

    number_rows = nrow(data)

    open("$DATA_FOLDER_TRANS/$DATASET_TRANS", "w") do file
        for index in 1:number_rows
            println("$index / $number_rows")
            record = data[index, :]
            api = record.api
            context = record.context

            if haskey(mapping, record.fqn)
                y_true = mapping[record.fqn]
                only_api = false

                if typeof(context) == Missing
                    context = ""
                    only_api = true
                end

                vector = obtain_vector(api_model, context_model, api, context, only_api, 20)
                write(file, "$(join(vector, ",")),$y_true\n")
            end
        end
    end
end

main()
