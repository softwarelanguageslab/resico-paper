package main

import (
	"fmt"

	"github.com/ynqa/wego/pkg/model/word2vec"
)

func main() {

	modelsPath := "/mansion/cavelazq/PhD/resico_sampled"
	NROW := 50 // This parameter should be changed accordingly to the number of rows sampled

	modelsPathSampled := fmt.Sprintf("%s/sample_%d", modelsPath, NROW)
	csvPath := fmt.Sprintf("/mansion/cavelazq/PhD/COSTER/data/sample_%d_data/contextsRESICO_%d_selected.csv", NROW, NROW)

	fmt.Println("Reading all elements in the sampled file ..")
	lines := getLinesFile(csvPath)
	apiLines, contextLines, fqnLines := getLinesByColumns(lines)
	fmt.Println("Done!")

	// Configure the model for the both, the APIs and the contexts
	const numberFeatures = 20
	fmt.Println("Creating the model to be trained on the corpus of data ...")
	modelW2V, errModel := word2vec.New(
		word2vec.BatchSize(1000),
		word2vec.Dim(numberFeatures),
		word2vec.Goroutines(20),
		word2vec.Iter(5),
		word2vec.MinCount(1),
		word2vec.Model(word2vec.Cbow),
		word2vec.Optimizer(word2vec.NegativeSampling),
		word2vec.Verbose(),
		word2vec.Window(5),
	)

	if errModel != nil {
		fmt.Println("Error creating the model", errModel)
		return
	}
	fmt.Println("Done!")

	fmt.Println("Transforming API elements and Contexts into vectors ...")
	APIsModel := fmt.Sprintf("%s/APIsW2V.model", modelsPathSampled)
	contextsModel := fmt.Sprintf("%s/ContextsW2V.model", modelsPathSampled)

	transformedAPIs := transformLines(apiLines, "api")
	trainSaveModel(modelW2V, transformedAPIs, APIsModel)

	transformedContexts := transformLines(contextLines, "context")
	trainSaveModel(modelW2V, transformedContexts, contextsModel)
	fmt.Println("Done!")

	// fmt.Println("Changing all input into vectors for further processing ...")
	// vectorsInput := averageModels(APIsModel, contextsModel, transformedAPIs, transformedContexts)
	// fmt.Println("Done!")

	fmt.Println("Transforming the FQNs into numbers, i.e., the classes that models need to predict ...")
	FQNsMappingPath := fmt.Sprintf("%s/mapping.txt", modelsPathSampled)
	FQNsTransformedPath := fmt.Sprintf("%s/fqns_transformed.txt", modelsPathSampled)

	transformFQNs(fqnLines, FQNsMappingPath, FQNsTransformedPath)
	fmt.Println("Done!")

	// fmt.Println("Joining the averaged vectors and the transformed FQNs into a single file for further processing ...")
	// data_path := fmt.Sprintf("%s/data_%d.csv", modelsPathSampled, NROW)
	// joinData(vectorsInput, transformedFQNs, data_path)
	// fmt.Println("Done!")
}
