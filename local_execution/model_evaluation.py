import os
import torch

from model import Net


def calculate_model_accuracies(test_dataloader, models_path):
    model_performance = {}
    net = Net()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for model_path in os.listdir(models_path):
            correct = 0
            total = 0
            if model_path.endswith("gitkeep"):
                continue

            net.load_state_dict(torch.load(os.path.join(models_path, model_path)))
            for data in test_dataloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            model_performance[model_path] = accuracy
            print(
                f"Accuracy of the model {model_path}: on the 7000 test images: {accuracy:.2f} %"
            )
    print("Ranking:", sorted(((v, k) for k, v in model_performance.items()), reverse=True))
