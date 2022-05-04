package main

import (
	"fmt"
)

func main() {
	models_path := "/mansion/cavelazq/PhD/resico_all"

	fmt.Println("Transforming API elements and Contexts into vectors ...")
	apis_model := fmt.Sprintf("%s/APIsW2V.model", models_path)
	context_model := fmt.Sprintf("%s/ContextsW2V.model", models_path)

	// w2v_input("api", apis_model)
	// w2v_input("context", context_model)
	// fmt.Println("Done!")

	fmt.Println("Averaging apis and context vectors ...")
	average_path := fmt.Sprintf("%s/averaged_vectors.txt", models_path)
	average_models(apis_model, context_model, average_path)
	fmt.Println("Done!")

	// fmt.Println("Transforming the FQNs into numbers, i.e., the classes that models need to predict ...")
	// fqns_mapping_path := fmt.Sprintf("%s/mapping.txt", models_path)
	fqns_transformed_path := fmt.Sprintf("%s/fqns_transformed.txt", models_path)

	// transform_fqns(apis_model, fqns_mapping_path, fqns_transformed_path)
	// fmt.Println("Done!")

	fmt.Println("Joining the averaged vectors and the transformed FQNs into a single file for further processing ...")
	data_path := fmt.Sprintf("%s/data.csv", models_path)
	join_data(average_path, fqns_transformed_path, data_path)
	fmt.Println("Done!")
}
