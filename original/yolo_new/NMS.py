from torchvision.ops import nms
import torch

# 가상 데이터
boxes = torch.rand((1000, 4)).cuda()  # 바운딩 박스
scores = torch.rand(1000).cuda()  # Confidence Score
threshold = 0.5  # IoU Threshold

# NMS 실행
keep_indices = nms(boxes, scores, threshold)

