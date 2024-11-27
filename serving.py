import sys, os, signal
import uvicorn
import pandas as pd
import json
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Response, Query
from fastapi.responses import FileResponse
from pathlib import Path
from datetime import datetime
from sse_starlette.sse import EventSourceResponse

from graphrag.config.load_config import load_config
from graphrag.config.resolve_path import resolve_paths
from graphrag.cli.query import _resolve_parquet_files
import graphrag.api as api

DEFAULT_COMMUNITY_LEVEL = 2
DEFAULT_RESPONSE_TYPE = "Multiple Paragraphs"

app = FastAPI()
svrs = []

class GraphRAGQueryRequest(BaseModel):
    root: str
    method: Literal["local", "global", "drift"]
    query: str
    streaming: bool
    graphrag_api_base: str
    graphrag_api_base_embedding: str
    graphrag_input_type: str

class GraphRAGQueryResponse(BaseModel):
    response: str


def extract_dir_name_datetime(folder_name):
    try:
        return datetime.strptime(folder_name, "%Y%m%d-%H%M%S")
    except ValueError:
        return None


# curl 'http://192.168.0.20:38062/get-graphml?index=test_zh&filename=summarized_graph.graphml'
@app.get("/get-graphml")
async def get_graphml(
    index: str = Query(..., description="graph index root"),
    filename: str = Query("summarized_graph.graphml", description="filename to get")):
    if not filename.endswith(".graphml"):
        raise HTTPException(status_code=404, detail="Only support graphml file")
    output_path = os.path.join("/workspace", index, "output")
    if os.path.exists(output_path):
        target_graphml = os.path.join(output_path, filename)
        print(f"target_graphml: {target_graphml}")
        if os.path.exists(target_graphml) and os.path.isfile(target_graphml):
            return FileResponse(target_graphml, media_type='application/xml')
        else:
            raise HTTPException(status_code=404, detail="File not found") 
    else:
        raise HTTPException(status_code=404, detail="File not found")


# curl -X POST -H 'Content-Type:application/json' -d '{"root":"/workspace/test_en", "method":"local", "streaming": false, "query":"why Musk is essential for OpenAI?", "graphrag_api_base":"http://192.168.0.20:38060/v1", "graphrag_api_base_embedding":"http://192.168.0.20:38060/v1", "graphrag_input_type":"text"}' http://192.168.0.20:38062/query
@app.post("/query")
async def query(req: GraphRAGQueryRequest):
    if req.root.strip() == "" or req.query.strip() == "":
        raise HTTPException(status_code=400, detail="Invalid request")
    if req.graphrag_api_base.strip() == "":
        raise HTTPException(status_code=400, detail="Param graphrag_api_base is required")
    
    os.environ["GRAPHRAG_API_BASE"] = req.graphrag_api_base
    os.environ["GRAPHRAG_API_BASE_EMBEDDING"] = req.graphrag_api_base_embedding
    os.environ["GRAPHRAG_INPUT_FILE_TYPE"] = req.graphrag_input_type
    
    print(f"[QUERY] request: {req}")

    if req.streaming:
        resp = run_query_streaming(
            root=Path(req.root),
            query=req.query,
            method=req.method
            )
        return EventSourceResponse(resp, media_type="text/event-stream")
    
    resp = await run_query(
        root=Path(req.root),
        query=req.query,
        method=req.method
        )
    if not resp:
        raise HTTPException(status_code=500, detail="Internal error")
    print(f"[QUERY] finished:\n{resp}\n")
    return Response(content=resp, media_type="text/plain")


async def run_query(
    root: Path,
    query: str,
    method: str,
    ):
    resp = None
    match method:
        case "local":
            resp = await run_local_search(
                root_dir=root,
                query=query,
            )
        case "global":
            resp = await run_global_search(
                root_dir=root,
                query=query,
            )
        case "drift":
            resp = await run_drift_search(
                root_dir=root,
                query=query,
            )
        case _:
            raise HTTPException(status_code=400, detail="Invalid request")
    return resp


async def run_query_streaming(
    root: Path,
    query: str,
    method: str,
    ):
    match method:
        case "local":
            async for chunk in run_local_search_streaming(root_dir=root, query=query):
                yield chunk
        case "global":
            async for chunk in run_global_search_streaming(root_dir=root, query=query):
                yield chunk
        case "drift":
            raise HTTPException(status_code=400, detail="Drift streaming not support")
        case _:
            raise HTTPException(status_code=400, detail="Invalid request")


async def build_local_request(root_dir: Path):
    root = root_dir.resolve()
    config = load_config(root, None)
    resolve_paths(config)

    dataframe_dict = await _resolve_parquet_files(
        root_dir=root_dir,
        config=config,
        parquet_list=[
            "create_final_nodes.parquet",
            "create_final_community_reports.parquet",
            "create_final_text_units.parquet",
            "create_final_relationships.parquet",
            "create_final_entities.parquet",
        ],
        optional_list=[
            "create_final_covariates.parquet",
        ],
    )
    final_nodes: pd.DataFrame = dataframe_dict["create_final_nodes"]
    final_community_reports: pd.DataFrame = dataframe_dict[
        "create_final_community_reports"
    ]
    final_text_units: pd.DataFrame = dataframe_dict["create_final_text_units"]
    final_relationships: pd.DataFrame = dataframe_dict["create_final_relationships"]
    final_entities: pd.DataFrame = dataframe_dict["create_final_entities"]
    final_covariates: pd.DataFrame | None = dataframe_dict["create_final_covariates"]

    return config, final_nodes, final_entities, final_community_reports, final_text_units,final_relationships, final_covariates


async def run_local_search(root_dir: Path, query: str):
    config, final_nodes, final_entities, final_community_reports, final_text_units,\
        final_relationships, final_covariates = await build_local_request(root_dir)
    
    # not streaming
    response, context_data = await api.local_search(
        config=config,
        nodes=final_nodes,
        entities=final_entities,
        community_reports=final_community_reports,
        text_units=final_text_units,
        relationships=final_relationships,
        covariates=final_covariates,
        community_level=DEFAULT_COMMUNITY_LEVEL,
        response_type=DEFAULT_RESPONSE_TYPE,
        query=query,
    )

    return response


async def run_local_search_streaming(root_dir: Path, query: str):
    config, final_nodes, final_entities, final_community_reports, final_text_units, final_relationships, final_covariates = await build_local_request(root_dir)

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
        community_level=DEFAULT_COMMUNITY_LEVEL,
        response_type=DEFAULT_RESPONSE_TYPE,
        query=query,
    ):
        if get_context_data:
            context_data = stream_chunk
            get_context_data = False
        else:
            full_response += stream_chunk
            yield(stream_chunk)
    yield("[DONE]")
    print(f"[QUERY] finished:\n{full_response}\n")


async def build_global_request(root_dir: Path):
    root = root_dir.resolve()
    config = load_config(root, None)
    resolve_paths(config)

    dataframe_dict = await _resolve_parquet_files(
        root_dir=root_dir,
        config=config,
        parquet_list=[
            "create_final_nodes.parquet",
            "create_final_entities.parquet",
            "create_final_communities.parquet",
            "create_final_community_reports.parquet",
        ],
        optional_list=[],
    )
    final_nodes: pd.DataFrame = dataframe_dict["create_final_nodes"]
    final_entities: pd.DataFrame = dataframe_dict["create_final_entities"]
    final_communities: pd.DataFrame = dataframe_dict["create_final_communities"]
    final_community_reports: pd.DataFrame = dataframe_dict[
        "create_final_community_reports"
    ]

    return config, final_nodes, final_entities, final_communities, final_community_reports


async def run_global_search(root_dir: Path, query: str,):
    config, final_nodes, final_entities, final_communities,\
        final_community_reports = await build_global_request(root_dir)
    
    # not streaming
    response, context_data = await api.global_search(
        config=config,
        nodes=final_nodes,
        entities=final_entities,
        communities=final_communities,
        community_reports=final_community_reports,
        community_level=DEFAULT_COMMUNITY_LEVEL,
        dynamic_community_selection=False,
        response_type=DEFAULT_RESPONSE_TYPE,
        query=query,
    )

    return response


async def run_global_search_streaming(root_dir: Path, query: str,):
    config, final_nodes, final_entities, final_communities,\
        final_community_reports = await build_global_request(root_dir)

    full_response = ""
    context_data = None
    get_context_data = True
    async for stream_chunk in api.global_search_streaming(
        config=config,
        nodes=final_nodes,
        entities=final_entities,
        communities=final_communities,
        community_reports=final_community_reports,
        community_level=DEFAULT_COMMUNITY_LEVEL,
        dynamic_community_selection=False,
        response_type=DEFAULT_RESPONSE_TYPE,
        query=query,
    ):
        if get_context_data:
            context_data = stream_chunk
            get_context_data = False
            
        else:
            full_response += stream_chunk
            yield(stream_chunk)
    yield("[DONE]")
    print(f"[QUERY] finished:\n{full_response}\n")


async def build_drift_search(root_dir: Path):
    root = root_dir.resolve()
    config = load_config(root, None)
    resolve_paths(config)

    dataframe_dict = await _resolve_parquet_files(
        root_dir=root_dir,
        config=config,
        parquet_list=[
            "create_final_nodes.parquet",
            "create_final_community_reports.parquet",
            "create_final_text_units.parquet",
            "create_final_relationships.parquet",
            "create_final_entities.parquet",
        ],
    )
    final_nodes: pd.DataFrame = dataframe_dict["create_final_nodes"]
    final_community_reports: pd.DataFrame = dataframe_dict[
        "create_final_community_reports"
    ]
    final_text_units: pd.DataFrame = dataframe_dict["create_final_text_units"]
    final_relationships: pd.DataFrame = dataframe_dict["create_final_relationships"]
    final_entities: pd.DataFrame = dataframe_dict["create_final_entities"]

    return config, final_nodes, final_entities, final_community_reports,\
        final_text_units, final_relationships


async def run_drift_search(root_dir: Path, query: str):
    config, final_nodes, final_entities, final_community_reports,\
        final_text_units, final_relationships = await build_drift_search(root_dir)

    # not streaming
    response, context_data = await api.drift_search(
        config=config,
        nodes=final_nodes,
        entities=final_entities,
        community_reports=final_community_reports,
        text_units=final_text_units,
        relationships=final_relationships,
        community_level=DEFAULT_COMMUNITY_LEVEL,
        query=query,
    )

    return response


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("[Main] shutting down...")
        for svr in svrs:
            svr.should_exit = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    conf = uvicorn.Config(app=app, host='0.0.0.0', port=80)
    server = uvicorn.Server(conf)
    svrs.append(server)
    print(f"[GraphRAG] start api server...")
    server.run()

    print("[Main] exited")
