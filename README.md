# graphrag-chinese

支持**中文**🇨🇳🇨🇳🇨🇳的 microsoft graphrag

来自 [microsoft/graphrag](https://github.com/microsoft/graphrag)

- 可以使用OpenAI协议兼容的中文大模型API，或者开源中文模型搭建的OpenAI协议兼容的API server
- 使用开源模型搭建API server可以显著降低索引构建成本

## 更新

- 🚀 `2025/04/03` **DRIFT query**支持streaming输出了，`cli.query`支持了`multi-index`查询，升级graphrag版本为 **v2.1.0**
- 🚀 `2025/01/14` 新增 **BASIC query** 模式，升级graphrag版本为 **v1.1.2**
- 🚀 `2024/11/27` 新增了 **DRIFT query** 模式，升级graphrag版本为 **v0.5.0**
- 🚀 `2024/09/13` 适配中文，graphrag版本为 **v0.3.3**

## 与原生 graphrag 的区别
- 定制的基于中文字符和标点符号的分词器
- 将 index 和 query 使用到的 prompt 翻译为中文

## tips
- 深度思考模型可能不适用

## 用法

### 配置env
新建`.env`文件到根目录，写入环境变量  
GRAPHRAG_API_BASE: OpenAI api 兼容的服务  
GRAPHRAG_API_BASE_EMBEDDING: OpenAI api 兼容的embedding服务  
GRAPHRAG_API_KEY: OpenAI api key  

### index
```
GRAPHRAG_INPUT_FILE_TYPE=text python index.py --root /working_root --input /input_files --lang chinese
```

--root: graphrag 工作目录  
--input: index 用到的输入文件夹  
--lang: 可选, 默认 english, 可用 chinese  
--update: 可选，更新已有的知识图谱索引  
GRAPHRAG_INPUT_FILE_TYPE: graphrag input 类型, text/csv  

### query
index 完成后, 使用 ```python serving.py``` 启动 FastAPI server，支持流式查询  
method: POST  
body:  
```
{
    "root": "/working_root",
    "method": "local",  # graphrag query method, can be local, global, drift or basic
    "query": "query prompt",
    "streaming": false  # streaming output
}
```
调用示例:  
```
curl -N -X POST -H 'Content-Type:application/json' -d '{"root":"/workspace/test", "method":"local", "query":"why Musk is essential for OpenAI?", "streaming": false}' 'http://192.168.0.20:38062/query'
```
