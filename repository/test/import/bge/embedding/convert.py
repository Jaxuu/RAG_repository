import torch
from safetensors.torch import save_file

# 你的本地权重路径
bin_path = r"D:\Develop\models\modelscope_cache\models\BAAI\bge-m3\pytorch_model.bin"
safetensors_path = r"D:\Develop\models\modelscope_cache\models\BAAI\bge-m3\model.safetensors"

print("正在读取 bin 权重文件...")
state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)

print("正在转换为 safetensors 格式...")
save_file(state_dict, safetensors_path)
print("转换成功！已生成 model.safetensors")