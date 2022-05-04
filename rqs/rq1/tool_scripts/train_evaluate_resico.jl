using CSV, DataFrames, DecisionTree, Printf, JLD2

const LeafOrNode{S, T} = Union{Leaf{T}, Node{S, T}}

"Applies `row_fun(X_row)::AbstractVector` to each row in X and returns a matrix containing the resulting vectors, stacked vertically."
function stack_function_results(row_fun::Function, X::AbstractMatrix, y::AbstractVector, top_k::Int)
    N = size(X, 1)
    out = Array{Int64}(undef, N, 2)
    for i in 1:N
        if i % 100_000 == 0
            @printf("%d / %d\n", i, N)
        end

        predictions = row_fun(X[i, :])
        y_true = y[i]
        top_k_predictions = get_topk(predictions, top_k)

        # In the case the true value is among the predictions
        if y_true in top_k_predictions
            out[i, 1] = y_true
            out[i, 2] = y_true
        # Otherwise just take any prediction, they are all wrong anyways
        else
            out[i, 1] = y_true
            out[i, 2] = top_k_predictions[1]
        end
    end
    return out
end

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
function save_evaluation(evaluation_matrix::Matrix{Int64}, top_k::Int, fold::Int)
    RESULTS_PATH = "results/intrinsic"
    results_folder = "$RESULTS_PATH/$top_k"

    if !isdir(results_folder)
        mkpath(results_folder)
    end

    results_path = "$results_folder/fold_$fold.txt"
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

function main()
    DATA_FOLDER = "strat_resico"
    println("Defining common structures ...")
    classifier = RandomForestClassifier(n_trees=10)

    top_ks = [1, 3, 5]
    println("Done!")

    for fold_index in 0:9
        println("Loading the data in fold $fold_index ...")
        data_train = DataFrame(CSV.File("$DATA_FOLDER/train_$fold_index.csv", header=false))
        data_test = DataFrame(CSV.File("$DATA_FOLDER/test_$fold_index.csv", header=false))
        println("Done!")

        println("Transforming the data files into a matrices ...")
        data_train_matrix = Matrix(data_train)
        data_test_matrix = Matrix(data_test)
        println("Done!")

        println("Dividing the training and testing data ...")
        # Training data
        X_train = convert(Array, data_train_matrix[:, 1:100])
        y_train = convert(Array, data_train_matrix[:, 101])

        # Testing data
        X_test = convert(Array, data_test_matrix[:, 1:100])
        y_test = convert(Array, data_test_matrix[:, 101])
        println("Done!")

        println("Training the classifier with the folded data ...")
        @time DecisionTree.fit!(classifier, X_train, y_train)
        println("Done!")

        println("Evaluating the trained classifier ...")
        for top_k in top_ks
            predictions = apply_forest_proba_matrix(classifier.ensemble, X_test, y_test, length(classifier.classes), top_k)
            save_evaluation(predictions, top_k, fold_index + 1)
        end
        println("Done!")
    end

    # # This last step is trained with the full data only and not with the stratified data
    # # The stratified data is used only for the purposes of a fair evaluation between folds
    
    # println("Loading the full data ...")
    # data = DataFrame(CSV.File("data.csv", header=false))

    # println("Training model with the full data ...")
    # classifier = RandomForestClassifier(n_trees=10)

    # matrix_data = Matrix(data)
    # X = convert(Array, matrix_data[:, 1:100])
    # y = convert(Array, matrix_data[:, 101])

    # @time fit!(classifier, X, y)
    # println("Done!")

    # println("Saving the trained model with the full data ...")
    # @time @save "$model_name.jld2" classifier
    # println("Done!")
end

main()
