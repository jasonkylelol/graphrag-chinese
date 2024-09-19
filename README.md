# graphrag-chinese

microsoft graphrag for **Chinese**🇨🇳🇨🇳🇨🇳 using local serving llm and embedding which has OpenAI compatible api  

Forked from [microsoft/graphrag](https://github.com/microsoft/graphrag), current using version is **v0.3.3**  

## Difference with original graphrag
- Customized text splitter for Chinese characters and punctuation marks
- Customized all index and query prompt for Chinese

## Usage

### index
```
GRAPHRAG_API_BASE=GRAPHRAG_API_BASE GRAPHRAG_API_KEY=GRAPHRAG_API_KEY GRAPHRAG_INPUT_FILE_TYPE=text python index.py --root /working_root --input /input_files --lang chinese
```

GRAPHRAG_API_BASE: OpenAI compatible api llm server  
GRAPHRAG_API_KEY: OpenAI compatible api key  
GRAPHRAG_INPUT_FILE_TYPE: graphrag input type, text or csv  

```
--root: working root directory  
--input: input files directory for index  
--lang: optional, default is english, chinese for specific document  
```

### query
when you finished index, use ```python serving.py``` to launch a server to do streaming query  
method: POST  
body:  
```
{
    "root": "/working_root",
    "method": "local",  # graphrag query method, can be local or global
    "query": "query prompt",
    "graphrag_api_base": "",  # OpenAI compatible api llm server
    "graphrag_input_type": "text"  # graphrag input type, text or csv
}
```

for example:  
```
curl -N -X POST -H 'Content-Type:application/json' -d '{"root":"/workspace/test", "method":"local", "query":"why Musk is essential for OpenAI?","graphrag_api_base":"https://open.bigmodel.cn/api/paas/v4/", "graphrag_input_type":"text"}' 'http://192.168.0.20:38062/query-streaming'
```
