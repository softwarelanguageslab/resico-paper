package main

import (
	"fmt"
	"os"
	"strconv"
)

func transformFQNs(fqns []string, mappingPath, transformPath string) {
	fmt.Println("Converting FQNs ...")

	f, err := os.Create(transformPath)
	if err != nil {
		fmt.Println(err)
	}
	defer f.Close()

	g, err := os.Create(mappingPath)
	if err != nil {
		fmt.Println(err)
	}
	defer g.Close()

	index_mapping := 0
	unique_types := make(map[string]int)

	for _, fqn := range fqns {
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

func joinData(vectors [][]float64, fqns []int, data_path string) {
	fmt.Println("Joining data and writing ...")
	dataFile, err := os.Create(data_path)
	if err != nil {
		fmt.Println(err)
	}
	defer dataFile.Close()

	for index, vector := range vectors {
		var vectorString string

		for _, number := range vector {
			vectorString += fmt.Sprintf("%f,", number)
		}

		fqn := fqns[index]
		dataLine := vectorString + strconv.Itoa(fqn)

		_, err := dataFile.WriteString(dataLine + "\n")
		if err != nil {
			fmt.Println(err)
		}
	}
}
