from lib.Network_Res2Net_GRA_NCD import *
#from models.ablation2 import *
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable

if __name__ == '__main__':
 from torchinfo import summary as torchinfo_summary
from thop import profile
input_data_1 = torch.randn(1, 3, 352, 352).cuda()
net = Network().cuda()
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
# 使用 thop 计算模型的 GFLOPs
flops, params = profile(net, inputs=(input_data_1,))
gflops = flops / 1e9 # 转换为十亿次浮点运算
print(f"parameters: {total_params}", f"GFLOPs: {gflops}")
