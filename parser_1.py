import ast

def extract_imports(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)
    
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)
    
    return list(imports)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python parser.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    dependencies_list = extract_imports(file_path)
    # Format the dependencies list into YAML format
    yaml_output = "dependencies:\n"
    for dependency in dependencies_list:
        yaml_output += f"  - {dependency}\n"

    # Print the YAML-formatted output
    print(yaml_output)
