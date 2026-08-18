from modelscope import snapshot_download

# 下载完整的 Reranker 模型仓库
model_dir = snapshot_download(
    model_id='BAAI/bge-reranker-large',
    local_dir=r'D:\Develop\models\modelscope_cache\models\BAAI\bge-reranker-large'
)
print(f"模型下载完成，存储路径：{model_dir}")