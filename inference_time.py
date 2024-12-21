import torch
import torchvision
import time
import tqdm
from torchsummary import summary


def calcGPUTime():
    device = 'cuda:0'
    model = torchvision.models.resnet18()
    model.to(device)
    model.eval()
    # summary(model, input_size=(3, 224, 224), device="cuda")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    num_iterations = 1000  # 迭代次数
    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)

    print('testing ...\n')
    total_forward_time = 0.0  # 使用time来测试
    # 记录开始时间
    start_event = time.time() * 1000
    with torch.no_grad():
        for _ in tqdm.tqdm(range(num_iterations)):
            start_forward_time = time.time()
            _ = model(dummy_input)
            end_forward_time = time.time()
            forward_time = end_forward_time - start_forward_time
            total_forward_time += forward_time * 1000  # 转换为毫秒

    # 记录结束时间
    end_event = time.time() * 1000

    elapsed_time = (end_event - start_event) / 1000.0  # 转换为秒
    fps = num_iterations / elapsed_time

    elapsed_time_ms = elapsed_time / (num_iterations * dummy_input.shape[0])

    avg_forward_time = total_forward_time / (num_iterations * dummy_input.shape[0])

    print(f"FPS: {fps}")
    print("elapsed_time_ms:", elapsed_time_ms * 1000)
    print(f"Avg Forward Time per Image: {avg_forward_time} ms")


if __name__ == "__main__":
    calcGPUTime()
