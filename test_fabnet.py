import sys
import os
import torch

fabnet_code_path = os.path.join(os.getcwd(), 'FAb-Net', 'FAb-Net', 'code')
if fabnet_code_path not in sys.path:
    sys.path.append(fabnet_code_path)

from models_multiview import FrontaliseModelMasks_wider

inner_nc = 256
num_additional_ids = 32

try:
    model = FrontaliseModelMasks_wider(3, inner_nc=inner_nc, num_additional_ids=num_additional_ids)
    weights_path = r"D:\Emotion\release_bmvc_fabnet\release\aflw_4views.pth"
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint['state_dict_model'])
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
