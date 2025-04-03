import sys, os, signal
import uvicorn
import pandas as pd
import json
import asyncio
from typing import List, Literal, Optional, Union, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Response, Query
from fastapi.responses import FileResponse
from pathlib import Path
from datetime import datetime
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

import graphrag.api as api
from graphrag.config.load_config import load_config
from graphrag.utils.api import create_storage_from_config
from graphrag.utils.storage import load_table_from_storage, storage_has_table
from graphrag.callbacks.noop_query_callbacks import NoopQueryCallbacks
from graphrag.config.enums import SearchMethod
from graphrag.config.models.graph_rag_config import GraphRagConfig

DEFAULT_COMMUNITY_LEVEL = 2
DEFAULT_RESPONSE_TYPE = "Multiple Paragraphs"

load_dotenv()

app = FastAPI()
svrs = []

class GraphRAGQueryRequest(BaseModel):
    root: str
    method: Literal["local", "global", "drift", "basic"]
    query: str
    streaming: bool

class GraphRAGQueryResponse(BaseModel):
    response: str

# curl 'http://192.168.0.20:38062/get-graphml?index=test_zh&filename=summarized_graph.graphml'
@app.get("/get-graphml")
async def get_graphml(
    index: str = Query(..., description="graph index root"),
    filename: str = Query("graph.graphml", description="filename to get")):
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


# curl -X POST -H 'Content-Type:application/json' -d '{"root":"/workspace/test_en", "method":"local", "streaming": false, "query":"why Musk is essential for OpenAI?"}' http://192.168.0.20:38062/query
@app.post("/query")
async def query(req: GraphRAGQueryRequest):
    if req.root.strip() == "" or req.query.strip() == "":
        raise HTTPException(status_code=400, detail="Invalid request")
    print(f"[QUERY] request: {req}")
    
    return await run_query(root=Path(req.root), query=req.query, method=SearchMethod(req.method), streaming=req.streaming)


async def run_query(
    root: Path,
    query: str,
    method: SearchMethod,
    streaming: bool,
    ):
    match method:
        case SearchMethod.LOCAL:
            return await run_local_search(root_dir=root, query=query, streaming=streaming)
        case SearchMethod.GLOBAL:
            return await run_global_search(root_dir=root, query=query, streaming=streaming)
        case SearchMethod.DRIFT:
            return await run_drift_search(root_dir=root, query=query, streaming=streaming)
        case SearchMethod.BASIC:
            return await run_basic_search(root_dir=root, query=query, streaming=streaming)
        case _:
            raise HTTPException(status_code=400, detail=f"Invalid request method: {method}")


async def build_local_request(root_dir: Path):
    root = root_dir.resolve()
    config = load_config(root)

    dataframe_dict = await _resolve_output_files(
        config=config,
        output_list=[
            "communities",
            "community_reports",
            "text_units",
            "relationships",
            "entities",
        ],
        optional_list=[
            "covariates",
        ],
    )

    final_communities: pd.DataFrame = dataframe_dict["communities"]
    final_community_reports: pd.DataFrame = dataframe_dict["community_reports"]
    final_text_units: pd.DataFrame = dataframe_dict["text_units"]
    final_relationships: pd.DataFrame = dataframe_dict["relationships"]
    final_entities: pd.DataFrame = dataframe_dict["entities"]
    final_covariates: pd.DataFrame | None = dataframe_dict["covariates"]

    return config, final_communities, final_community_reports, final_text_units,\
        final_relationships, final_entities, final_covariates


async def run_local_search(root_dir: Path, query: str, streaming: bool):
    config, final_communities, final_community_reports, final_text_units,\
        final_relationships, final_entities, final_covariates = await build_local_request(root_dir)
    
    if streaming:
        async def streaming_search():
            full_response = ""
            context_data = {}

            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context

            callbacks = NoopQueryCallbacks()
            callbacks.on_context = on_context
            
            async for stream_chunk in api.local_search_streaming(
                config=config,
                entities=final_entities,
                communities=final_communities,
                community_reports=final_community_reports,
                text_units=final_text_units,
                relationships=final_relationships,
                covariates=final_covariates,
                community_level=DEFAULT_COMMUNITY_LEVEL,
                response_type=DEFAULT_RESPONSE_TYPE,
                query=query,
                callbacks=[callbacks],
            ):
                full_response += stream_chunk
                yield(stream_chunk)
            yield("[DONE]")
        
        streaming_resp = streaming_search()
        return EventSourceResponse(streaming_resp, media_type="text/event-stream")

    response, context_data = await api.local_search(
        config=config,
        entities=final_entities,
        communities=final_communities,
        community_reports=final_community_reports,
        text_units=final_text_units,
        relationships=final_relationships,
        covariates=final_covariates,
        community_level=DEFAULT_COMMUNITY_LEVEL,
        response_type=DEFAULT_RESPONSE_TYPE,
        query=query,
    )
    return Response(content=response, media_type="text/plain")


async def build_global_request(root_dir: Path):
    root = root_dir.resolve()
    config = load_config(root)

    dataframe_dict = await _resolve_output_files(
        config=config,
        output_list=[
            "entities",
            "communities",
            "community_reports",
        ],
        optional_list=[],
    )

    final_entities: pd.DataFrame = dataframe_dict["entities"]
    final_communities: pd.DataFrame = dataframe_dict["communities"]
    final_community_reports: pd.DataFrame = dataframe_dict["community_reports"]

    return config, final_entities, final_communities, final_community_reports


async def run_global_search(root_dir: Path, query: str, streaming: bool):
    config, final_entities, final_communities,\
        final_community_reports = await build_global_request(root_dir)
    
    if streaming:
        async def streaming_search():
            full_response = ""
            context_data = {}

            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context

            callbacks = NoopQueryCallbacks()
            callbacks.on_context = on_context

            async for stream_chunk in api.global_search_streaming(
                config=config,
                entities=final_entities,
                communities=final_communities,
                community_reports=final_community_reports,
                community_level=DEFAULT_COMMUNITY_LEVEL,
                dynamic_community_selection=False,
                response_type=DEFAULT_RESPONSE_TYPE,
                query=query,
                callbacks=[callbacks],
            ):
                full_response += stream_chunk
                yield(stream_chunk)
            yield("[DONE]")

        streaming_resp = streaming_search()
        return EventSourceResponse(streaming_resp, media_type="text/event-stream")

    response, context_data = await api.global_search(
        config=config,
        entities=final_entities,
        communities=final_communities,
        community_reports=final_community_reports,
        community_level=DEFAULT_COMMUNITY_LEVEL,
        dynamic_community_selection=False,
        response_type=DEFAULT_RESPONSE_TYPE,
        query=query,
    )

    return response


async def build_drift_request(root_dir: Path):
    root = root_dir.resolve()
    config = load_config(root)

    dataframe_dict = await _resolve_output_files(
        config=config,
        output_list=[
            "communities",
            "community_reports",
            "text_units",
            "relationships",
            "entities",
        ],
    )
    final_communities: pd.DataFrame = dataframe_dict["communities"]
    final_community_reports: pd.DataFrame = dataframe_dict["community_reports"]
    final_text_units: pd.DataFrame = dataframe_dict["text_units"]
    final_relationships: pd.DataFrame = dataframe_dict["relationships"]
    final_entities: pd.DataFrame = dataframe_dict["entities"]

    return config, final_communities, final_entities, final_community_reports,\
        final_text_units, final_relationships


async def run_drift_search(root_dir: Path, query: str, streaming: bool):
    config, final_communities, final_entities, final_community_reports,\
        final_text_units, final_relationships = await build_drift_request(root_dir)

    if streaming:
        async def streaming_search():
            full_response = ""
            context_data = {}

            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context

            callbacks = NoopQueryCallbacks()
            callbacks.on_context = on_context

            async for stream_chunk in api.drift_search_streaming(
                config=config,
                entities=final_entities,
                communities=final_communities,
                community_reports=final_community_reports,
                text_units=final_text_units,
                relationships=final_relationships,
                community_level=DEFAULT_COMMUNITY_LEVEL,
                response_type=DEFAULT_RESPONSE_TYPE,
                query=query,
                callbacks=[callbacks],
            ):
                full_response += stream_chunk
                yield(stream_chunk)
            yield("[DONE]")

        streaming_resp = streaming_search()
        return EventSourceResponse(streaming_resp, media_type="text/event-stream")

    response, context_data = await api.drift_search(
        config=config,
        entities=final_entities,
        communities=final_communities,
        community_reports=final_community_reports,
        text_units=final_text_units,
        relationships=final_relationships,
        community_level=DEFAULT_COMMUNITY_LEVEL,
        response_type=DEFAULT_RESPONSE_TYPE,
        query=query,
    )

    return response


async def build_basic_request(root_dir: Path):
    root = root_dir.resolve()
    config = load_config(root)

    dataframe_dict = await _resolve_output_files(
        config=config,
        output_list=[
            "text_units",
        ],
    )
    final_text_units: pd.DataFrame = dataframe_dict["text_units"]

    return config, final_text_units


async def run_basic_search(root_dir: Path, query: str, streaming: bool):
    config, final_text_units = await build_basic_request(root_dir)

    if streaming:
        async def streaming_search():
            full_response = ""
            context_data = {}

            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context

            callbacks = NoopQueryCallbacks()
            callbacks.on_context = on_context

            async for stream_chunk in api.basic_search_streaming(
                config=config,
                text_units=final_text_units,
                query=query,
            ): 
                full_response += stream_chunk
                yield(stream_chunk)
            yield("[DONE]")

        streaming_resp = streaming_search()
        return EventSourceResponse(streaming_resp, media_type="text/event-stream")

    response, context_data = await api.basic_search(
        config=config,
        text_units=final_text_units,
        query=query,
    )
    
    return response


async def _resolve_output_files(
    config: GraphRagConfig,
    output_list: list[str],
    optional_list: list[str] | None = None,
) -> dict[str, Any]:
    """Read indexing output files to a dataframe dict."""
    dataframe_dict = {}

    # Loading output files for multi-index search
    if config.outputs:
        dataframe_dict["multi-index"] = True
        dataframe_dict["num_indexes"] = len(config.outputs)
        dataframe_dict["index_names"] = config.outputs.keys()
        for output in config.outputs.values():
            storage_obj = create_storage_from_config(output)
            for name in output_list:
                if name not in dataframe_dict:
                    dataframe_dict[name] = []
                df_value = await load_table_from_storage(name=name, storage=storage_obj)
                dataframe_dict[name].append(df_value)

            # for optional output files, do not append if the dataframe does not exist
            if optional_list:
                for optional_file in optional_list:
                    if optional_file not in dataframe_dict:
                        dataframe_dict[optional_file] = []
                    file_exists = await storage_has_table(optional_file, storage_obj)
                    if file_exists:
                        df_value = await load_table_from_storage(
                                name=optional_file, storage=storage_obj
                            )
                        dataframe_dict[optional_file].append(df_value)
        return dataframe_dict
    # Loading output files for single-index search
    dataframe_dict["multi-index"] = False
    storage_obj = create_storage_from_config(config.output)
    for name in output_list:
        df_value = await load_table_from_storage(name=name, storage=storage_obj)
        dataframe_dict[name] = df_value

    # for optional output files, set the dict entry to None instead of erroring out if it does not exist
    if optional_list:
        for optional_file in optional_list:
            file_exists = await storage_has_table(optional_file, storage_obj)
            if file_exists:
                df_value = await load_table_from_storage(name=optional_file, storage=storage_obj)
                dataframe_dict[optional_file] = df_value
            else:
                dataframe_dict[optional_file] = None
    return dataframe_dict


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
