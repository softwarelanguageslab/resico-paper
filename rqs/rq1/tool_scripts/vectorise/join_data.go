package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func join_data(vectors_path, fqns_path, data_path string) {
	vectors_file, err := os.Open(vectors_path)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("Successfully opened vectors file ...")
	defer vectors_file.Close()

	fqns_file, err := os.Open(fqns_path)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("Successfully opened fqns file ...")
	defer fqns_file.Close()

	fmt.Println("Reading all vectors ...")
	scanner_vector := bufio.NewScanner(vectors_file)
	scanner_vector.Split(bufio.ScanLines)
	var vector_lines []string

	for scanner_vector.Scan() {
		vector_lines = append(vector_lines, scanner_vector.Text())
	}
	fmt.Println("Done!")

	fmt.Println("Reading all fqns ...")
	scanner_fqn := bufio.NewScanner(fqns_file)
	scanner_fqn.Split(bufio.ScanLines)
	var fqn_lines []string

	for scanner_fqn.Scan() {
		fqn_lines = append(fqn_lines, scanner_fqn.Text())
	}
	fmt.Println("Done!")

	fmt.Println("Joining data and writing ...")
	data_file, err := os.Create(data_path)
	if err != nil {
		fmt.Println(err)
	}
	defer data_file.Close()

	for index, vector := range vector_lines {
		// Note: Vectors end with a space, thus, that last space is replace by a comma
		// and therefore is not necessary to append yet another comma
		vector_modified := strings.ReplaceAll(vector, " ", ",")

		fqn_index := fqn_lines[index]
		data_line := vector_modified + fqn_index

		_, err := data_file.WriteString(data_line + "\n")
		if err != nil {
			fmt.Println(err)
		}
	}

}
