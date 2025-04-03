import sys, os
import argparse
import shutil
import time
import asyncio
from pathlib import Path
import logging
from dotenv import load_dotenv
import yaml
from string import Template

from graphrag.logger.types import LoggerType
from graphrag.cli.index import index_cli, update_cli
from graphrag.config.enums import IndexingMethod

load_dotenv()

# GRAPHRAG_INPUT_FILE_TYPE=text python index.py --root /workspace/test_zh --input /workspace/zh --lang chinese
# GRAPHRAG_INPUT_FILE_TYPE=text python index.py --root /workspace/test_en --input /workspace/en

def replace_settings(root_dir: str):
    input_file_type = os.getenv("GRAPHRAG_INPUT_FILE_TYPE", "text")

    settings_path = os.path.join(root_dir, "settings.yaml")
    with open(settings_path, "r", encoding="utf-8") as file:
        yaml_content = file.read()

    template = Template(yaml_content)
    updated_content = template.safe_substitute(GRAPHRAG_INPUT_FILE_TYPE=input_file_type)

    data = yaml.safe_load(updated_content)

    with open(settings_path, "w", encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--root",
        help="Working index directory",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--input",
        help="Path of input files",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--lang",
        help="Language for graphrag",
        type=str,
    )
    parser.add_argument(
        "--update",
        help="Update an existing knowledge graph index",
        action="store_true",
    )
    args = parser.parse_args()

    stime = time.time()

    workdir = args.root
    os.makedirs(workdir, exist_ok=True)

    if not args.update:
        if args.lang == "chinese":
            print(f"Using language: Chinese")
            shutil.copytree("template_zh", workdir, dirs_exist_ok=True)
        else:
            print(f"Using default language: English")
            shutil.copytree("template", workdir, dirs_exist_ok=True)

    replace_settings(workdir)
    
    shutil.copytree(args.input, os.path.join(workdir, "input"), dirs_exist_ok=True)

    if args.update:
        print("Method: update")
        update_cli(
            root_dir=Path(workdir),
            verbose=False,
            memprofile=False,
            cache=True,
            logger=LoggerType(LoggerType.PRINT),
            config_filepath=None,
            skip_validation=False,
            output_dir=None,
            method=IndexingMethod.Standard,
        )
    else:
        print("Method: index")
        index_cli(
            root_dir=Path(workdir),
            verbose=False,
            memprofile=False,
            cache=True,
            logger=LoggerType(LoggerType.PRINT),
            config_filepath=None,
            dry_run=False,
            skip_validation=False,
            output_dir=None,
            method=IndexingMethod.Standard,
        )
