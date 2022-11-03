using Word2Vec, CSV, DataFrames

function obtain_vector(model_api::WordVectors, model_context::WordVectors, api::AbstractString, context::AbstractString, vectors_dim::Int64)
    context_divided = split(context, " ")

    vector_api = zeros(vectors_dim)

    if api in model_api.vocab
        vector_api = get_vector(model_api, api)
    end

    vector_sum_context = zeros(vectors_dim)
    found = 0
    map(context_divided) do context_word
        if context_word in model_context.vocab
            vector_context_word = get_vector(model_context, context_word)
            vector_sum_context += vector_context_word
            found += 1
        end
    end

    if found > 0
        vector_context = vector_sum_context / found
        vector_averaged = (vector_api + vector_context) / 2
    else
        vector_averaged = vector_api
    end

    vector_not_nan = filter(value -> !isnan(value), vector_averaged)

    if length(vector_not_nan) != length(vector_averaged)
        println("NaN values appeared!")
        exit(1)
    end

    return vector_averaged
end

function transform_data(api_word::AbstractString, context::AbstractString)
    transformed_api = replace(api_word, "|" => ".")
    transformed_context = replace(context, "|" => " ")

    return (transformed_api, transformed_context)
end

function main()
    NROW = 50
    DIMESION_VECTORS = 20
    dataPATH = "/mansion/cavelazq/PhD/resico_sampled/sample_$(NROW)"

    println("Loading the word2vec models ...")
    api_model = wordvectors("$dataPATH/APIsW2V.model")
    context_model = wordvectors("$dataPATH/ContextsW2V.model")
    println("Done!")

    println("Reading the sampled data ...")
    CSV_PATH = "/mansion/cavelazq/PhD/COSTER/data/sample_$(NROW)_data/contextsRESICO_$(NROW)_selected.csv"
    data = DataFrame(CSV.File(CSV_PATH))
    println("Done!")

    println("Reading the transformed FQNs ...")
    lines_fqns = readlines("$dataPATH/fqns_transformed.txt")
    println("Done!")

    number_rows = nrow(data)

    println("Transforming the sampled data ...")
    open("$dataPATH/data.csv", "w") do file
        for index in 1:number_rows
            record = data[index, :]
            api = record.api
            context = record.context

            if index % 10_000 == 0
                println("$index / $number_rows")
            end

            if typeof(context) == Missing
                context = ""
            end

            transformed_data = transform_data(api, context)

            averaged_vector = obtain_vector(api_model, context_model, transformed_data[1], transformed_data[2], DIMESION_VECTORS)
            rounded_vector = averaged_vector
            write(file, "$(join(rounded_vector, ",")),$(lines_fqns[index])\n")
        end
    end
    println("Done!")
end

main()
