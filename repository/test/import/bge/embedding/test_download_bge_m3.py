import os
import shutil
from modelscope import snapshot_download

# 1. 先下载到临时目录
temp_dir = snapshot_download(
    model_id='BAAI/bge-m3',
    cache_dir='D:\\Develop\\models\\modelscope_cache'
)

# 2. 目标路径
target_dir = 'D:\\Develop\\models\\modelscope_cache\\models\\BAAI\\bge-m3'

# 3. 删除已存在的目标目录（如果有）
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

# 4. 移动文件到目标路径
shutil.copytree(temp_dir, target_dir)
print(f"✅ 模型已移动到: {target_dir}")