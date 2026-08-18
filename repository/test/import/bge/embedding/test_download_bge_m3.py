from modelscope.hub.file_download import model_file_download

model_file_download(
    model_id='BAAI/bge-m3',
    file_path='model.safetensors',
    local_dir='D:\\Develop\\models\\modelscope_cache\\models\\BAAI\\bge-m3'
)

"""

bge-m3 原生的嵌入模型【使用起来麻烦一点】计算---存储到其它向量数据库中(redis)
milvus----集成了特别多的模型（bge-m3嵌入模型）
"""