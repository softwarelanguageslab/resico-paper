package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/ynqa/wego/pkg/search"
)

func read_apis_fqns() ([]string, []string) {
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

	var elementsAPIs, elementsFQNs []string

	fmt.Println("Transforming lines ...")
	for _, line := range csvLines {
		elementsAPIs = append(elementsAPIs, strings.ReplaceAll(line[0], "|", "."))
		elementsFQNs = append(elementsFQNs, line[2])
	}

	return elementsAPIs, elementsFQNs
}

func convert_fqns(apis, fqns []string, search_apis *search.Searcher, mapping_path, tranformed_path string) {
	f, err := os.Create(tranformed_path)
	if err != nil {
		fmt.Println(err)
	}
	defer f.Close()

	g, err := os.Create(mapping_path)
	if err != nil {
		fmt.Println(err)
	}
	defer g.Close()

	index_mapping := 0
	unique_types := make(map[string]int)

	for index, api := range apis {
		fmt.Printf("%d / %d\n", index+1, len(apis))

		// Vectors
		api_emb, _ := search_apis.Items.Find(api)
		api_vector := api_emb.Vector

		// FQN
		fqn := fqns[index]

		// Default vector
		api_vector_scaled := make([]float64, 100)

		if len(api_vector) == len(api_vector_scaled) {

			_, found := unique_types[fqn]

			if !found {
				index_mapping += 1
				unique_types[fqn] = index_mapping
			}

			value := unique_types[fqn]
			_, err := f.WriteString(strconv.Itoa(value) + "\n")
			if err != nil {
				fmt.Println(err)
			}

			mapping := fmt.Sprintf("%d,%s", value, fqn)
			_, err = g.WriteString(mapping + "\n")
			if err != nil {
				fmt.Println(err)
			}
		}
	}
}

func transform_fqns(path_model_apis, path_mapping, transform_path string) {
	fmt.Println("Reading trained models ...")
	embeddings_apis := get_embeddings(path_model_apis)
	fmt.Println("Done !")

	fmt.Println("Making searchers ...")
	searcher_apis, _ := search.New(embeddings_apis...)
	fmt.Println("Done !")

	fmt.Println("Reading the FQNs ...")
	apis, fqns := read_apis_fqns()
	fmt.Println("Done!")

	fmt.Println("Converting FQNs ...")
	convert_fqns(apis, fqns, searcher_apis, path_mapping, transform_path)
}
