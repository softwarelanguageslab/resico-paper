package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strings"

	"github.com/ynqa/wego/pkg/embedding"
	"github.com/ynqa/wego/pkg/search"
	"gonum.org/v1/gonum/floats"
)

func get_embeddings(path_model string) embedding.Embeddings {
	input, err := os.Open(path_model)
	if err != nil {
		fmt.Println(err)
	}
	defer input.Close()

	embs, err := embedding.Load(input)
	if err != nil {
		fmt.Println(err)
	}

	return embs
}

// func get_maximum_emb(embeddings embedding.Embeddings) float64 {
// 	max_matrix := -math.MaxFloat64

// 	for _, embedding := range embeddings {
// 		vector_embedding := embedding.Vector
// 		max_vector := floats.Max(vector_embedding)

// 		if max_vector > max_matrix {
// 			max_matrix = max_vector
// 		}
// 	}

// 	return max_matrix
// }

func read_elements() ([]string, []string) {
	csv_path := "/mansion/cavelazq/PhD/COSTER/data/contextsRESICO.csv"

	csvFile, err := os.Open(csv_path)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("Successfully opened CSV file ...")
	defer csvFile.Close()

	fmt.Println("Reading all lines ...")
	csvLines, err := csv.NewReader(csvFile).ReadAll()
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("Done!")

	var elementsAPIs, elementsContexts []string

	fmt.Println("Transforming lines ...")
	for _, line := range csvLines {
		elementsAPIs = append(elementsAPIs, strings.ReplaceAll(line[0], "|", "."))
		elementsContexts = append(elementsContexts, strings.ReplaceAll(line[1], "|", " "))
	}

	return elementsAPIs, elementsContexts
}

func average_vector(vector []float64, number_vectors float64) []float64 {
	averaged_vector := make([]float64, 100)

	for i, number := range vector {
		averaged_vector[i] = number / number_vectors
	}

	return averaged_vector
}

func average_vectors(apis, contexts []string, search_apis, search_contexts *search.Searcher, path_data string) {
	f, err := os.Create(path_data)
	if err != nil {
		fmt.Println(err)
	}
	defer f.Close()

	for index, api := range apis {
		if index > 0 {
			fmt.Printf("%d / %d\n", index, len(apis))
			// Words
			context := contexts[index]

			// Vectors
			api_emb, _ := search_apis.Items.Find(api)
			api_vector := api_emb.Vector

			// The contexts need to be divided and then each word mapped to a vector
			context_divided := strings.Split(context, " ")
			sum_context_vectors := make([]float64, 100)

			for _, context_word := range context_divided {
				context_emb, _ := search_contexts.Items.Find(context_word)
				context_vector := context_emb.Vector

				if len(context_vector) == 100 {
					floats.AddTo(sum_context_vectors, sum_context_vectors, context_vector)
				}
			}

			context_vector := average_vector(sum_context_vectors, float64(len(context_divided)))

			// // Scale vectors
			// api_vector_scaled := make([]float64, 100)

			// if len(api_vector) == len(api_vector_scaled) {
			// floats.ScaleTo(api_vector_scaled, 1/scale_apis, api_vector)

			// context_vector_scaled := make([]float64, 100)
			// floats.ScaleTo(context_vector_scaled, 1/scale_contexts, context_vector)

			// Average of the vectors
			sum_vector := make([]float64, 100)
			floats.AddTo(sum_vector, api_vector, context_vector)
			averaged_vectors := average_vector(sum_vector, 2)
			var vector_string string

			for _, number := range averaged_vectors {
				vector_string += fmt.Sprintf("%f ", number)
			}

			_, err := f.WriteString(vector_string + "\n")
			if err != nil {
				fmt.Println(err)
			}
		}
	}
	fmt.Println("Write Finished")
}

func average_models(path_model_apis, path_model_contexts, path_transformed_data string) {
	fmt.Println("Reading trained models ...")
	embeddings_apis := get_embeddings(path_model_apis)
	embeddings_contexts := get_embeddings(path_model_contexts)
	fmt.Println("Done !")

	fmt.Println("Making searchers ...")
	searcher_apis, _ := search.New(embeddings_apis...)
	searcher_contexts, _ := search.New(embeddings_contexts...)
	fmt.Println("Done!")

	// fmt.Println("Finding maximum values in the matrices ...")
	// max_matrix_apis := get_maximum_emb(embeddings_apis)
	// max_matrix_contexts := get_maximum_emb(embeddings_contexts)
	// fmt.Println("Done!")

	fmt.Println("Reading the elements to transform them ...")
	elementsAPIS, elementsContext := read_elements()
	fmt.Println("Done!")

	fmt.Println("Averaging the vectors ...")
	average_vectors(elementsAPIS, elementsContext, searcher_apis, searcher_contexts, path_transformed_data)
}
