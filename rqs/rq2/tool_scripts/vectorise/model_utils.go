package main

import (
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/ynqa/wego/pkg/embedding"
	"github.com/ynqa/wego/pkg/model"
	"github.com/ynqa/wego/pkg/model/modelutil/vector"
	"github.com/ynqa/wego/pkg/search"
	"gonum.org/v1/gonum/floats"
)

func getEmbeddings(pathModel string) embedding.Embeddings {
	input, err := os.Open(pathModel)
	if err != nil {
		fmt.Println(err)
	}
	defer func(input *os.File) {
		err := input.Close()
		if err != nil {
			return
		}
	}(input)

	embeddings, err := embedding.Load(input)
	if err != nil {
		fmt.Println(err)
	}

	return embeddings
}

func getMaximumEmbedding(embeddings embedding.Embeddings) float64 {
	maxMatrix := -math.MaxFloat64

	for _, e := range embeddings {
		vectorEmbedding := e.Vector
		maxVector := floats.Max(vectorEmbedding)

		if maxVector > maxMatrix {
			maxMatrix = maxVector
		}
	}
	return maxMatrix
}

func averageVector(vector []float64, numberVectors float64) []float64 {
	averagedVector := make([]float64, 100)

	for i, number := range vector {
		averagedVector[i] = number / numberVectors
	}

	return averagedVector
}

func averageVectors(apis, contexts []string, searchApis, searchContexts *search.Searcher, scaleAPIs, scaleContexts float64) [][]float64 {
	vectors := make([][]float64, len(apis))

	for index, api := range apis {
		if index%10000 == 0 {
			fmt.Printf("Vector %d / %d\n", index, len(apis))
		}

		// Words
		context := contexts[index]

		// Vectors
		apiEmb, _ := searchApis.Items.Find(api)
		apiVector := apiEmb.Vector

		// The contexts need to be divided and then each word mapped to a vector
		contextDivided := strings.Split(context, " ")
		sumContextVectors := make([]float64, 100)

		for _, contextWord := range contextDivided {
			contextEmb, _ := searchContexts.Items.Find(contextWord)
			contextVector := contextEmb.Vector

			if len(contextVector) == 100 {
				floats.AddTo(sumContextVectors, sumContextVectors, contextVector)
			}
		}

		contextVector := averageVector(sumContextVectors, float64(len(contextDivided)))

		// Scale vectors
		apiVectorScaled := make([]float64, 100)

		if len(apiVectorScaled) == len(apiVector) {
			floats.ScaleTo(apiVectorScaled, 1/scaleAPIs, apiVector)
		}

		contextVectorScaled := make([]float64, 100)
		floats.ScaleTo(contextVectorScaled, 1/scaleContexts, contextVector)

		// Average of the vectors
		sumVector := make([]float64, 100)
		floats.AddTo(sumVector, apiVectorScaled, contextVectorScaled)
		averagedVectors := averageVector(sumVector, 2)

		vectors[index] = averagedVectors
	}
	fmt.Println("Done!")
	return vectors
}

func averageModels(pathModelApis, pathModelContexts string, elementsAPIs, elementsContext []string) [][]float64 {
	fmt.Println("Reading trained models ...")
	embeddingsApis := getEmbeddings(pathModelApis)
	embeddingsContexts := getEmbeddings(pathModelContexts)
	fmt.Println("Done !")

	fmt.Println("Making searchers ...")
	searcherApis, _ := search.New(embeddingsApis...)
	searcherContexts, _ := search.New(embeddingsContexts...)
	fmt.Println("Done!")

	fmt.Println("Finding maximum values in the matrices ...")
	maxMatrixAPIs := getMaximumEmbedding(embeddingsApis)
	maxMatrixContexts := getMaximumEmbedding(embeddingsContexts)
	fmt.Println("Done!")

	fmt.Println("Averaging the vectors ...")
	return averageVectors(elementsAPIs, elementsContext, searcherApis, searcherContexts, maxMatrixAPIs, maxMatrixContexts)
}

func trainSaveModel(w2vArch model.Model, selectedLines []string, modelPath string) {
	elementsJoined := strings.NewReader(strings.Join(selectedLines, "\n"))
	errTraining := w2vArch.Train(elementsJoined)
	if errTraining != nil {
		fmt.Println("Error in the training", errTraining)
		return
	}

	fmt.Println("Done!")
	fmt.Println("Saving the w2v trained model ...")
	fileModel, errCreating := os.Create(modelPath)
	if errCreating != nil {
		fmt.Println("Error creating the file to save the model", errCreating)
		return
	}

	errSaving := w2vArch.Save(fileModel, vector.Agg)
	if errSaving != nil {
		fmt.Println("Error saving the model to a file", errSaving)
		return
	}
	errClosing := fileModel.Close()
	if errClosing != nil {
		fmt.Println("Error closing the file", errClosing)
		return
	}
}
