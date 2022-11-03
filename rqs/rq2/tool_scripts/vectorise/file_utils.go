package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strings"
)

func getLinesFile(pathFile string) [][]string {
	file, err := os.Open(pathFile)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("Successfully opened file passed as argument ...")
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			return
		}
	}(file)

	fmt.Println("Reading all lines ...")
	csvReader := csv.NewReader(file)
	lines, err := csvReader.ReadAll()
	if err != nil {
		fmt.Println(err)
	}

	return lines
}

func getLinesByColumns(lines [][]string) ([]string, []string, []string) {
	var apis []string
	var contexts []string
	var fqns []string

	fmt.Println("Getting the values by column ...")
	for index, line := range lines {
		if index == 0 {
			continue
		}
		apis = append(apis, line[0])
		contexts = append(contexts, line[1])
		fqns = append(fqns, line[2])
	}

	return apis, contexts, fqns
}

func transformLines(lines []string, column string) []string {
	var elementsTransformed []string

	fmt.Println("Transforming lines ...")
	for _, line := range lines {
		if column == "api" {
			elementsTransformed = append(elementsTransformed, strings.ReplaceAll(line, "|", "."))
		} else if column == "context" {
			elementsTransformed = append(elementsTransformed, strings.ReplaceAll(line, "|", " "))
		}
	}
	fmt.Println("Done!")

	return elementsTransformed
}
