import numpy as np
import torch
import torch.nn.functional as F


def calculate_entropy(logins, batch=False):
    """
    compute info_entropy

    param:
    probabilities: list of entropy

    return:
    entropy: result of entropy, float number
    """
    if batch:
        entropy_list = []
        for row in logins:
            activated_logins = F.relu(row)
            probabilities = F.tanh(activated_logins)
            probabilities = probabilities + 1e-10
            if torch.is_tensor(probabilities):
                probabilities = probabilities.numpy()
            entropy = -np.sum(probabilities * np.log2(probabilities))
            entropy_list.append(entropy)
        return entropy_list
    else:
        probabilities = F.tanh(logins)
        if torch.is_tensor(probabilities):
            probabilities = probabilities.numpy()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy


if __name__ == '__main__':
    # 示例：假设有一个二进制随机变量，概率分布为 [0.2, 0.8]
    random_sequence = torch.rand(768)
    probabilities_binary = random_sequence
    entropy_binary = calculate_entropy(probabilities_binary)
    print(f"Binary Entropy: {entropy_binary}")

    # 示例：假设有一个骰子，概率分布为 [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
    probabilities_dice = torch.tensor([0.01]*768)
    entropy_dice = calculate_entropy(probabilities_dice)
    print(f"Dice Entropy: {entropy_dice}")