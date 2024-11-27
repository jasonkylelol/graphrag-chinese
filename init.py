from graphrag.cli.initialize import initialize_project_at
import sys

if __name__ == "__main__":
    root = sys.argv[1]
    initialize_project_at(path=root)