from graphrag.cli.initialize import initialize_project_at
import sys, argparse
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--root",
        help="Working index directory",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--force",
        help="Force initialization even if the project already exists",
        type=bool,
        default=False,
    )
    args = parser.parse_args()

    initialize_project_at(path=args.root, force=args.force)