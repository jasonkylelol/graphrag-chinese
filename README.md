# graphrag-chinese

支持**中文**🇨🇳🇨🇳🇨🇳的 microsoft graphrag，可以使用兼容OpenAI api的本地开源模型或者其他api  

来自 [microsoft/graphrag](https://github.com/microsoft/graphrag), 使用版本为 **v0.3.3**  

## 与原生 graphrag 的区别
- 定制的基于中文字符和标点符号的分词器
- 将 index 和 query 使用到的 prompt 翻译为中文

## 用法

### index
```
GRAPHRAG_API_BASE=GRAPHRAG_API_BASE GRAPHRAG_API_KEY=GRAPHRAG_API_KEY GRAPHRAG_INPUT_FILE_TYPE=text python index.py --root /working_root --input /input_files --lang chinese
```

GRAPHRAG_API_BASE: OpenAI api 兼容的服务  
GRAPHRAG_API_KEY: OpenAI api key  
GRAPHRAG_INPUT_FILE_TYPE: graphrag input 类型, text/csv  

```
--root: graphrag 工作目录  
--input: index 用到的输入文件夹  
--lang: 可选, 默认 english, 可用 chinese  
```

### query
index 完成后, 使用 ```python serving.py``` 启动 FastAPI server，支持流式查询  
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
可能需要设置api key: GRAPHRAG_API_KEY  
调用示例:  
```
curl -N -X POST -H 'Content-Type:application/json' -d '{"root":"/workspace/test", "method":"local", "query":"why Musk is essential for OpenAI?","graphrag_api_base":"https://open.bigmodel.cn/api/paas/v4/", "graphrag_input_type":"text"}' 'http://192.168.0.20:38062/query-streaming'
```
