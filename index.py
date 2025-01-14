import sys, os
import argparse
import shutil
import time
import asyncio
from pathlib import Path
import logging

from graphrag.utils.cli import redact
import graphrag.api as api
from graphrag.logger.base import ProgressLogger
from graphrag.logger.factory import LoggerFactory, LoggerType
from graphrag.config.enums import CacheType
from graphrag.config.load_config import load_config
from graphrag.config.resolve_path import resolve_paths

# GRAPHRAG_API_BASE=http://192.168.0.20:38060/v1 GRAPHRAG_API_BASE_EMBEDDING=http://192.168.0.20:38060/v1 GRAPHRAG_INPUT_FILE_TYPE=text python index.py --root /workspace/test_zh --input /workspace/zh --lang chinese

# GRAPHRAG_API_BASE=http://192.168.0.20:38060/v1 GRAPHRAG_API_BASE_EMBEDDING=http://192.168.0.20:38060/v1 GRAPHRAG_INPUT_FILE_TYPE=text python index.py --root /workspace/test_en --input /workspace/en

import warnings
warnings.filterwarnings("ignore", message=".*NumbaDeprecationWarning.*")

log = logging.getLogger(__name__)

def _logger(logger: ProgressLogger):
    def info(msg: str, verbose: bool = False):
        log.info(msg)
        if verbose:
            logger.info(msg)

    def error(msg: str, verbose: bool = False):
        log.error(msg)
        if verbose:
            logger.error(msg)

    def success(msg: str, verbose: bool = False):
        log.info(msg)
        if verbose:
            logger.success(msg)

    return info, error, success


def _register_signal_handlers(logger: ProgressLogger):
    import signal

    def handle_signal(signum, _):
        # Handle the signal here
        logger.info(f"Received signal {signum}, exiting...")  # noqa: G004
        logger.dispose()
        for task in asyncio.all_tasks():
            task.cancel()
        logger.info("All tasks cancelled. Exiting...")

    # Register signal handlers for SIGINT and SIGHUP
    signal.signal(signal.SIGINT, handle_signal)

    if sys.platform != "win32":
        signal.signal(signal.SIGHUP, handle_signal)


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

    stime = time.time()

    workdir = os.path.join("/workspace", args.root)
    # os.makedirs(workdir, exist_ok=True)

    if args.lang == "chinese":
        shutil.copytree("template_zh", workdir, dirs_exist_ok=True)
    else:
        print(f"using default language: English")
        shutil.copytree("template", workdir, dirs_exist_ok=True)
    
    shutil.copytree(args.input, os.path.join(workdir, "input"), dirs_exist_ok=True)

    logger = LoggerType
    progress_logger = LoggerFactory().create_logger(logger)
    info, error, success = _logger(progress_logger)

    root = Path(workdir).resolve()
    config = load_config(root, None)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    resolve_paths(config, run_id)

    print(f"\nStarting pipeline run for: {run_id}\n")
    print(f"Using default configuration: {redact(config.model_dump())}")

    _register_signal_handlers(progress_logger)

    outputs = asyncio.run(
        api.build_index(
            config=config,
            run_id=run_id,
            is_resume_run=False,
            memory_profile=False,
            progress_logger=progress_logger,
        )
    )

    encountered_errors = any(
        output.errors and len(output.errors) > 0 for output in outputs
    )
    progress_logger.stop()

    etime_str = time.strftime("%Y%m%d-%H%M%S")
    print(f"Indexer finished at {etime_str} cost: {(time.time() - stime):.2f}")

    sys.exit(1 if encountered_errors else 0)
