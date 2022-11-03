from sys import argv

def fqn_in_names(fqn: str, key_names: list):
    for name in key_names:
        if fqn.startswith(name):
            return True, name
    return False, ""


def read_mapping():
    lines = []

    with open("/Users/kmilo/Dev/PhD/RESICO_new/rq1/results/models/resico/mapping.txt") as f:
        while True:
            line = f.readline()

            if not line:
                break
            else:
                line = line.strip()
                lines.append(line)
    
    unique_lines = list(set(lines))
    dictionary_lines = {}

    for line in unique_lines:
        divided_line = line.split(",")
        dictionary_lines[int(divided_line[0]) - 1] = divided_line[1]

    return dictionary_lines


if __name__ == '__main__':
    if len(argv) < 2:
        print("Not enough parameters, please enter the name of the model and the dataset!")
    elif len(argv) == 2:
        dataset = argv[1]
        LIBRARIES_FOLDER = "/Users/kmilo/Dev/PhD/RESICO_new/discussion/external_data"

        # It is only considered the results for Top-1
        RESULTS_FILE_COSTER = "/Users/kmilo/Dev/PhD/RESICO_new/rq2/results/coster/1/{}.txt".format(dataset)
        RESULTS_FILE_RESICO = "/Users/kmilo/Dev/PhD/RESICO_new/rq2/results/resico/knn/1/{}.txt".format(dataset)
        mapping = read_mapping()

        all_true_fqns = []
        all_predicted_fqns_coster = []
        all_predicted_fqns_resico = []

        with open(RESULTS_FILE_COSTER) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                else:
                    line = line.strip()
                    divided_line = line.split(",")

                    true_fqn = divided_line[0]
                    pred_fqn = divided_line[1]

                    all_true_fqns.append(true_fqn)
                    all_predicted_fqns_coster.append(pred_fqn)
        
        with open(RESULTS_FILE_RESICO) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                else:
                    line = line.strip()
                    divided_line = line.split(",")
                    pred_fqn = divided_line[1]
                    pred_fqn = mapping[int(pred_fqn)]
                    all_predicted_fqns_resico.append(pred_fqn)
        
        patterns = []
        success_coster = []
        failures_coster = []

        success_resico = []
        failures_resico = []

        dictionary_name = {
            "java": "JDK",
            "char": "JDK",
            "com.google.gwt": "GWT",
            "com.thoughtworks.xstream": "XStream",
            "org.hibernate": "Hibernate", 
            "org.apache.log4j": "Apache-Log4J", 
            "org.dom4j": "Dom4J", 
            "org.json": "Org-JSON",
            "org.apache.commons": "Apache-Commons", 
            "org.apache.http": "Apache-Http",
            "org.joda.time": "Joda-Time", 
            "org.apache.struts": "Apache-Struts", 
            "org.ksoap2": "KSoap2",
            "com.google.gson": "Gson",
            "org.springframework": "Spring",
            "org.jfree": "JFree",
            "org.jsoup": "JSoup",
            "org.junit": "JUnit",
            "com.google.common": "Guava",
            "org.apache.poi": "Apache-POI",
            "com.jcraft.jsch": "JCraft",
            "org.w3c.dom": "W3C",
            "com.fasterxml.jackson": "Jackson",
            "org.yaml.snakeyaml": "SnakeYAML"
        }

        for (true_fqn, pred_fqn_coster, pred_fqn_resico) in zip(all_true_fqns, all_predicted_fqns_coster, all_predicted_fqns_resico):
            in_name, pattern = fqn_in_names(true_fqn, list(dictionary_name.keys()))
            
            if pattern != '':
                name_pattern = dictionary_name[pattern]

            if in_name:
                if name_pattern not in patterns:
                    patterns.append(name_pattern)

                    if true_fqn == pred_fqn_coster:
                        success_coster.append(1)
                        failures_coster.append(0)
                    else:
                        success_coster.append(0)
                        failures_coster.append(1)

                    if true_fqn == pred_fqn_resico:
                        success_resico.append(1)
                        failures_resico.append(0)
                    else:
                        success_resico.append(0)
                        failures_resico.append(1)
                else:
                    index_pattern = patterns.index(name_pattern)

                    if true_fqn == pred_fqn_coster:
                        success_coster[index_pattern] += 1
                    else:
                        failures_coster[index_pattern] += 1

                    if true_fqn == pred_fqn_resico:
                        success_resico[index_pattern] += 1
                    else:
                        failures_resico[index_pattern] += 1
            else:
                if true_fqn not in patterns:
                    patterns.append(true_fqn)

                    if true_fqn == pred_fqn_coster:
                        success_coster.append(1)
                        failures_coster.append(0)
                    else:
                        success_coster.append(0)
                        failures_coster.append(1)

                    if true_fqn == pred_fqn_resico:
                        success_resico.append(1)
                        failures_resico.append(0)
                    else:
                        success_resico.append(0)
                        failures_resico.append(1)
                else:
                    index_pattern = patterns.index(true_fqn)

                    if true_fqn == pred_fqn_coster:
                        success_coster[index_pattern] += 1
                    else:
                        failures_coster[index_pattern] += 1

                    if true_fqn == pred_fqn_resico:
                        success_resico[index_pattern] += 1
                    else:
                        failures_resico[index_pattern] += 1

        # Check the patterns
        with open("{}/{}/libraries.csv".format(LIBRARIES_FOLDER, dataset.lower()), "w") as f:
            f.write("Library,Total,Success,Failure,Model\n")
            for (pattern, succ, fail) in zip(patterns, success_coster, failures_coster):
                f.write("{},{},{},{},COSTER\n".format(pattern, succ + fail, succ, fail))

            f.write("\n")
            for (pattern, succ, fail) in zip(patterns, success_resico, failures_resico):
                f.write("{},{},{},{},RESICO-KNN\n".format(pattern, succ + fail, succ, fail))
