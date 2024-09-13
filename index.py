import sys, os
import argparse
import shutil
import time
import asyncio
from pathlib import Path

from graphrag.index.cli import index_cli, _redact
from graphrag.index.progress.load_progress_reporter import load_progress_reporter
from graphrag.index.api import build_index
from graphrag.config import load_config

# GRAPHRAG_API_BASE=http://192.168.0.20:38063/v1 GRAPHRAG_INPUT_FILE_TYPE=text python index.py --root /workspace/test --input /workspace/chn/ --lang chinese

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
    args = parser.parse_args()

    workdir = os.path.join("/workspace", args.root)
    # os.makedirs(workdir, exist_ok=True)

    if args.lang == "chinese":
        shutil.copytree("template_zh", workdir, dirs_exist_ok=True)
    else:
        print(f"using default language: English")
        shutil.copytree("template", workdir, dirs_exist_ok=True)
    
    shutil.copytree(args.input, os.path.join(workdir, "input"), dirs_exist_ok=True)

    progress_reporter = load_progress_reporter("print")
    stime = time.time()
    run_id = time.strftime("%Y%m%d-%H%M%S")

    root = Path(workdir).resolve()
    config = load_config(root, None, run_id)
    print(f"\nStarting pipeline run for: {run_id}\n")
    print(f"Using default configuration: {_redact(config.model_dump())}")

    outputs = asyncio.run(
        build_index(
            config=config,
            run_id=run_id,
            memory_profile=False,
            progress_reporter=progress_reporter,
            emit=None,
        )
    )

    encountered_errors = any(
        output.errors and len(output.errors) > 0 for output in outputs
    )
    progress_reporter.stop()

    etime_str = time.strftime("%Y%m%d-%H%M%S")
    print(f"Indexer finished at {etime_str} cost: {(time.time() - stime):.2f}")
    sys.exit(1 if encountered_errors else 0)
