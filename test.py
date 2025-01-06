from model.yolo import YoloModel
import torch

def main() -> None:
    device = 'cuda'
    
    model = YoloModel().to(device)

    test_tensor = torch.rand(size=(1, 3, 448, 448)).to(device)

    print(f'--> input tensor size:  {test_tensor.shape}\n')

    output_tensor = model(test_tensor)

    print(f'\n--> output tensor size: {output_tensor.shape}')
    
    print(model)
    
if __name__ == '__main__':
    main()