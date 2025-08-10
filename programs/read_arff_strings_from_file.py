def read_arff_strings_from_file(file_path):
    arff_strings = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.endswith('.arff'):
                arff_strings.append(line)
    return arff_strings


file_path = 'C:\\Users\\praneeth\\Desktop\\proj_iwnn\\docs\\nets_log.txt'  # Replace with the actual file path
arff_strings = read_arff_strings_from_file(file_path)

for arff_string in arff_strings:
    print(arff_string)