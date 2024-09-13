import sys, os, signal
import uvicorn
import pandas as pd
import json
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Response, Query
from fastapi.responses import FileResponse, PlainTextResponse
from graphrag.query.cli import run_global_search, run_local_search
from graphrag.config import load_config
from graphrag.query import api
from datetime import datetime
from sse_starlette.sse import EventSourceResponse
from pathlib import Path

DEFAULT_COMMUNITY_LEVEL = 2
DEFAULT_RESPONSE_TYPE = "Multiple Paragraphs"

app = FastAPI()
svrs = []

class GraphRAGQueryRequest(BaseModel):
    root: str
    method: Literal["local", "global"]
    query: str
    graphrag_api_base: str
    graphrag_input_type: str

class GraphRAGQueryResponse(BaseModel):
    response: str


def extract_dir_name_datetime(folder_name):
    try:
        return datetime.strptime(folder_name, "%Y%m%d-%H%M%S")
    except ValueError:
        return None


# curl 'http://192.168.0.20:38062/get-graphml?index=test1&filename=summarized_graph.graphml
@app.get("/get-graphml")
def get_graphml(
    index: str = Query(..., description="graph index root"),
    filename: str = Query("summarized_graph.graphml", description="filename to get")):
    output_path = os.path.join("/workspace", index, "output")
    if os.path.exists(output_path):
        subfolders = [f.path for f in os.scandir(output_path) if f.is_dir()]
        latest_subfolder = max(
            (folder for folder in subfolders if extract_dir_name_datetime(os.path.basename(folder))),
            key=lambda folder: extract_dir_name_datetime(os.path.basename(folder))
        )
        target_graphml = os.path.join(latest_subfolder, f"artifacts/{filename}")
        print(f"target_graphml: {target_graphml}")
        if os.path.exists(target_graphml) and os.path.isfile(target_graphml):
            return FileResponse(target_graphml, media_type='application/xml')
        else:
            raise HTTPException(status_code=404, detail="File not found") 
    else:
        raise HTTPException(status_code=404, detail="File not found")


# curl -X POST -H 'Content-Type:application/json' -d '{"root":"/workspace/test", "method":"local", "query":"why Musk is essential for OpenAI?","graphrag_api_base":"https://open.bigmodel.cn/api/paas/v4/", "graphrag_input_type":"text"}' http://192.168.0.20:38062/query
@app.post("/query",  response_class=PlainTextResponse)
def query(req: GraphRAGQueryRequest):
    if req.root.strip() == "" or req.query.strip() == "":
        raise HTTPException(status_code=400, detail="Invalid request")
    if req.graphrag_api_base.strip() == "":
        raise HTTPException(status_code=400, detail="Param graphrag_api_base is required")
    
    os.environ["GRAPHRAG_API_BASE"] = req.graphrag_api_base
    os.environ["GRAPHRAG_INPUT_FILE_TYPE"] = req.graphrag_input_type
    
    print(f"[query] request: {req}")
    if req.method == "local":
        resp, context_data = run_local_search(
            config_filepath=None,
            data_dir=None,
            root_dir=req.root,
            community_level=DEFAULT_COMMUNITY_LEVEL,
            response_type=DEFAULT_RESPONSE_TYPE,
            streaming=False,
            query=req.query,
        )
    elif req.method == "global":
        resp, context_data = run_global_search(
            config_filepath=None,
            data_dir=None,
            root_dir=req.root,
            community_level=DEFAULT_COMMUNITY_LEVEL,
            response_type=DEFAULT_RESPONSE_TYPE,
            streaming=False,
            query=req.query,
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid request")
    
    return resp


# curl -N -X POST -H 'Content-Type:application/json' -d '{"root":"/workspace/test", "method":"local", "query":"why Musk is essential for OpenAI?","graphrag_api_base":"https://open.bigmodel.cn/api/paas/v4/", "graphrag_input_type":"text"}' 'http://192.168.0.20:38062/query-streaming'
@app.post("/query-streaming")
def query(req: GraphRAGQueryRequest):
    if req.root.strip() == "" or req.query.strip() == "":
        raise HTTPException(status_code=400, detail="Invalid request")
    if req.graphrag_api_base.strip() == "":
        raise HTTPException(status_code=400, detail="Param graphrag_api_base is required")
    
    os.environ["GRAPHRAG_API_BASE"] = req.graphrag_api_base
    os.environ["GRAPHRAG_INPUT_FILE_TYPE"] = req.graphrag_input_type
    
    print(f"[query] request: {req}")
    if req.method == "local":
        resp = local_query(
            root_dir=req.root,
            community_level=DEFAULT_COMMUNITY_LEVEL,
            response_type=DEFAULT_RESPONSE_TYPE,
            query=req.query,
        )
        return EventSourceResponse(resp, media_type="text/event-stream")
    elif req.method == "global":
        resp = global_query(
            root_dir=req.root,
            community_level=DEFAULT_COMMUNITY_LEVEL,
            response_type=DEFAULT_RESPONSE_TYPE,
            query=req.query,
        )
        return EventSourceResponse(resp, media_type="text/event-stream")
    else:
        raise HTTPException(status_code=400, detail="Invalid request")


async def local_query(
    root_dir: str,
    community_level: int,
    response_type: str,
    query: str,
):
    root = Path(root_dir).resolve()
    config = load_config(root, None)

    data_path = Path(config.storage.base_dir).resolve()

    final_nodes = pd.read_parquet(data_path / "create_final_nodes.parquet")
    final_community_reports = pd.read_parquet(
        data_path / "create_final_community_reports.parquet"
    )
    final_text_units = pd.read_parquet(data_path / "create_final_text_units.parquet")
    final_relationships = pd.read_parquet(
        data_path / "create_final_relationships.parquet"
    )
    final_entities = pd.read_parquet(data_path / "create_final_entities.parquet")
    final_covariates_path = data_path / "create_final_covariates.parquet"
    final_covariates = (
        pd.read_parquet(final_covariates_path)
        if final_covariates_path.exists()
        else None
    )

    full_response = ""
    context_data = None
    get_context_data = True
    async for stream_chunk in api.local_search_streaming(
        config=config,
        nodes=final_nodes,
        entities=final_entities,
        community_reports=final_community_reports,
        text_units=final_text_units,
        relationships=final_relationships,
        covariates=final_covariates,
        community_level=community_level,
        response_type=response_type,
        query=query,
    ):
        if get_context_data:
            context_data = stream_chunk
            get_context_data = False
            print(f"context_data:\n{json.dumps(context_data, ensure_ascii=False)}\n")
        else:
            full_response += stream_chunk
            yield stream_chunk
    yield "[DONE]"
    print(f"full_response:\n{full_response}\n")


async def global_query(
    root_dir: str,
    community_level: int,
    response_type: str,
    query: str,
):
    root = Path(root_dir).resolve()
    config = load_config(root, None)

    data_path = Path(config.storage.base_dir).resolve()

    final_nodes: pd.DataFrame = pd.read_parquet(
        data_path / "create_final_nodes.parquet"
    )
    final_entities: pd.DataFrame = pd.read_parquet(
        data_path / "create_final_entities.parquet"
    )
    final_community_reports: pd.DataFrame = pd.read_parquet(
        data_path / "create_final_community_reports.parquet"
    )

    full_response = ""
    context_data = None
    get_context_data = True
    async for stream_chunk in api.global_search_streaming(
        config=config,
        nodes=final_nodes,
        entities=final_entities,
        community_reports=final_community_reports,
        community_level=community_level,
        response_type=response_type,
        query=query,
    ):
        if get_context_data:
            context_data = stream_chunk
            get_context_data = False
            print(f"context_data:\n{json.dumps(context_data, ensure_ascii=False)}\n")
        else:
            full_response += stream_chunk
            yield stream_chunk
    yield "[DONE]"
    print(f"full_response:\n{full_response}\n")


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("[Main] shutting down...")
        for svr in svrs:
            svr.should_exit = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    config = uvicorn.Config(app=app, host='0.0.0.0', port=80)
    server = uvicorn.Server(config)
    svrs.append(server)
    print(f"[GraphRAG] start api server...")
    server.run()

    print("[Main] exited")
