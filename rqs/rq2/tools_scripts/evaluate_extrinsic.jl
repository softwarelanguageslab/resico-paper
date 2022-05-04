using DataFrames, CSV, DecisionTree, JLD2, Word2Vec

"Given an ensemble and a matrix, it computes the probabilities for each of the classes to be most likely the prediction."
function apply_forest_proba_matrix(forest::DecisionTree.Ensemble{S, T}, X::AbstractMatrix{S}, y::AbstractVector{S}, n_labels::Int, top_k::Int) where {S, T}
    stack_function_results(row->apply_forest_proba_vector(forest, row, n_labels), X, y, top_k)
end

"Given a forest and a vector, computes the probabilities for each of the classes to be most likely the prediction."
function apply_forest_proba_vector(forest::DecisionTree.Ensemble{S, T}, features::AbstractVector{S}, n_labels::Int) where {S, T}
    votes = [apply_tree(tree, features) for tree in forest.trees]
    return compute_probabilities_mod(votes, n_labels)
end

"Obtains the normalised weights for each of the classes in the data."
function compute_probabilities_mod(votes::AbstractVector, number_labels::Int)
    max_value = Int(round(findmax(votes)[1]))

    if max_value > number_labels
        number_labels = max_value
    end

    counts = zeros(Float64, number_labels)
    for vote in votes
        counts[Int(round(vote))] += 1.0
    end
    return counts / sum(counts)
end

"Through partial sort of the results, it obtains the highest top-k predictions for the evaluation."
function get_topk(vector::AbstractVector, k::Int)
    return partialsortperm(vector, 1:k, rev=true)
end

"Saves the results of the evaluation to external files which can be analysed later by metrics in any library."
function save_evaluation(evaluation_matrix::Matrix{Int64}, dataset_name::String, top_k::Int)
    RESULTS_PATH = "results/extrinsic"
    results_folder = "$RESULTS_PATH/$top_k"

    results_path = "$results_folder/$dataset_name.txt"
    n_rows = size(evaluation_matrix, 1)

    open(results_path, "w") do file 
        for i in 1:n_rows
            vector_eval = evaluation_matrix[i, :]
            true_value = vector_eval[1]
            predicted_value = vector_eval[2]

            write(file, "$true_value,$predicted_value\n")
        end
    end
end

function obtain_vector(model_api::WordVectors, model_context::WordVectors, api_word::AbstractString, context::AbstractString, only_api::Bool=false)
    transformed_api = replace(api_word, "|" => ".")
    transformed_context = replace(context, "|" => " ")

    vector_api = zeros(100)
    vector_sum_context = zeros(100)

    try
        vector_api = get_vector(model_api, transformed_api) 
    catch
    end

    if only_api
        return vector_api
    end

    context_divided = split(transformed_context, " ")

    for context_word in context_divided
        vector_context_word = zeros(100)

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
    DATASET = "StatType-SO"
    DATA_FOLDER = "datasets-resico"
    
    println("Loading CSV file ...")
    data = DataFrame(CSV.File("$DATA_FOLDER/$DATASET.csv"))
    println("Done!")

    println("Loading the classifier ...")
    classifier = load("randomForest.jld2", "classifier")
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

    top_ks = [1, 3, 5]
    number_rows = nrow(data)

    for top_k in top_ks
        println("Evaluating top-$top_k ...")
        out = Array{Int64}(undef, 0, 2)
        i = 0

        for index in 1:number_rows
            println("$index / $number_rows")
            record = data[index, :]
            api = record.api
            context = record.context

            if haskey(mapping, record.fqn)
                i += 1
                y_true = mapping[record.fqn]
                only_api = false

                if typeof(context) == Missing
                    context = ""
                    only_api = true
                end

                vector = obtain_vector(api_model, context_model, api, context, only_api)
                predictions = apply_forest_proba_vector(classifier.ensemble, vector, length(classifier.classes))
                top_k_predictions = get_topk(predictions, top_k)

                # In the case the true value is among the predictions
                if y_true in top_k_predictions
                    vector_correct_pred = [y_true y_true]
                    out = vcat(out, vector_correct_pred)
                # Otherwise just take any prediction, they are all wrong anyways
                else
                    vector_incorect_pred = [y_true top_k_predictions[1]]
                    out = vcat(out, vector_incorect_pred) 
                end
            end
        end

        println("Saving evaluation ...")
        save_evaluation(out, DATASET, top_k)
        println("Done!")
        println()
    end
end

main()
