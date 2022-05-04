using Word2Vec, CSV, DataFrames

function obtain_averaged_vector(model_api::WordVectors, model_context::WordVectors, api_word::AbstractString, context::AbstractString, only_api::Bool=false)
    transformed_api = replace(api_word, "|" => ".")
    transformed_context = replace(context, "|" => " ")

    vector_api = zeros(100)
    vector_sum_context = zeros(100)

    if transformed_api in model_api.vocab
        vector_api = get_vector(model_api, transformed_api)
    end

    if only_api
        return vector_api
    end

    context_divided = split(transformed_context, " ")

    map(context_divided) do context_word
        # vector_context_word = zeros(100)
        vector_context_word = get_vector(model_context, context_word)

        vector_sum_context += vector_context_word
    end

    vector_context = vector_sum_context / length(context_divided)
    vector_averaged = (vector_api + vector_context) / 2

    return vector_averaged
end

function obtain_vector(model_api::WordVectors, model_context::WordVectors, api::AbstractString, context::AbstractString)
    context_divided = split(context, " ")

    vector_api = zeros(100)

    if api in model_api.vocab
        vector_api = get_vector(model_api, api)
    end

    vector_sum_context = zeros(100)
    found = 0
    map(context_divided) do context_word
        if context_word in model_context.vocab
            vector_context_word = get_vector(model_context, context_word)
            vector_sum_context += vector_context_word
            found += 1
        end
    end

    vector_context = vector_sum_context / found
    vector_averaged = (vector_api + vector_context) / 2

    return vector_averaged
end

function transform_data(model_api::WordVectors, api_word::AbstractString, context::AbstractString, only_api::Bool)
    transformed_api = replace(api_word, "|" => ".")
    transformed_context = replace(context, "|" => " ")

    if transformed_api in model_api.vocab && !only_api
        return (transformed_api, transformed_context)
    end
end

function main()
    println("Loading the word2vec models ...")
    api_model = wordvectors("APIsW2V.model")
    context_model = wordvectors("ContextsW2V.model")
    println("Done!")

    println("Reading the full data ...")
    CSV_PATH = "/mansion/cavelazq/PhD/COSTER/data/contextsRESICO.csv"
    data = DataFrame(CSV.File(CSV_PATH))
    println("Done!")

    println("Reading the transformed FQNs ...")
    lines_fqns = readlines("fqns_transformed.txt")
    println("Done!")

    number_rows = nrow(data)

    open("data.csv", "w") do file
        for index in 1:number_rows
            record = data[index, :]
            api = record.api
            context = record.context

            if index % 200_000 == 0
                println("$index / $number_rows")
            end

            only_api = false

            if typeof(context) == Missing
                context = ""
                only_api = true
            end

            transformed_data = transform_data(api_model, api, context, only_api)

            if typeof(transformed_data) != Nothing
                averaged_vector = obtain_vector(api_model, context_model, transformed_data[1], transformed_data[2])

                write(file, "$(join(averaged_vector, ",")),$(lines_fqns[index])\n")
            end
        end
    end
end

main()
