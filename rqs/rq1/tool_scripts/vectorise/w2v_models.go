package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strings"

	"github.com/ynqa/wego/pkg/model/modelutil/vector"
	"github.com/ynqa/wego/pkg/model/word2vec"
)

func w2v_input(column string, model_path string) {
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

	var elementsTransformed []string

	fmt.Println("Transforming lines ...")
	for _, line := range csvLines {
		if column == "api" {
			elementsTransformed = append(elementsTransformed, strings.ReplaceAll(line[0], "|", "."))
		} else if column == "context" {
			elementsTransformed = append(elementsTransformed, strings.ReplaceAll(line[1], "|", " "))
		}
	}
	fmt.Println("Done!")

	fmt.Println("Fitting the w2v model ...")

	model, err := word2vec.New(
		word2vec.BatchSize(10000),
		word2vec.Dim(100),
		word2vec.Goroutines(20),
		word2vec.Iter(5),
		word2vec.MinCount(1),
		word2vec.Model(word2vec.Cbow),
		word2vec.Optimizer(word2vec.NegativeSampling),
		word2vec.Verbose(),
		word2vec.Window(5),
	)

	if err != nil {
		fmt.Println("Error creating the model", err)
		return
	}

	elementsJoined := strings.NewReader(strings.Join(elementsTransformed, "\n"))
	model.Train(elementsJoined)

	fmt.Println("Done!")

	fmt.Println("Saving the w2v model ...")

	fileModel, err := os.Create(model_path)
	if err != nil {
		fmt.Println("Error saving the model", err)
		return
	}

	model.Save(fileModel, vector.Agg)
	fileModel.Close()

	fmt.Println("Done!")
}
