import copy

import torch
from torch import optim, nn
from torch.cuda.amp import autocast, GradScaler

from lib.decorators.timing import record_time
from models.model import Model


def vectorize_net(model_weight):
    vectorized_weight = []
    for key, value in model_weight.items():
        flattened_tensor = torch.flatten(value)
        vectorized_weight.append(flattened_tensor)
    vectorized_tensor = torch.cat(vectorized_weight, dim=0)
    return vectorized_tensor


def root_train(root_net, root_train_loader, root_optimizer, criterion, device):
    root_net.train()
    scaler = GradScaler()

    num_epochs = 3
    root_avg_loss = 0
    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, data in enumerate(root_train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            root_optimizer.zero_grad()

            with autocast():
                outputs = root_net(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(root_optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(root_train_loader)
        root_avg_loss += avg_loss

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    root_avg_loss /= num_epochs

    return root_net.state_dict(), root_avg_loss


@record_time
def fltrust_module(model_weights_list, global_model_weights, root_train_loader, model_name, device):
    root_model = Model(model_name)
    root_net = root_model.get_model()
    root_net.to(device)
    root_net.load_state_dict(copy.deepcopy(global_model_weights))

    root_optimizer = optim.Adam(root_net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    root_model_trained_weight, root_avg_loss = root_train(root_net, root_train_loader, root_optimizer, criterion,
                                                          device)

    print("root model is being trained")
    print("root model average loss is: ", root_avg_loss)

    # 深拷贝全局模型权重
    pre_global_model_weights = copy.deepcopy(global_model_weights)

    # 计算根模型的更新
    root_update = copy.deepcopy(root_model_trained_weight)
    for key in root_update.keys():
        root_update[key] = root_update[key] - pre_global_model_weights[key]

    # 计算用户模型的权重更新
    user_model_update_list = copy.deepcopy(model_weights_list)
    for user_model_update in user_model_update_list:
        for key in pre_global_model_weights.keys():
            user_model_update[key] = user_model_update[key] - pre_global_model_weights[key]

    # 计算 Trust Score for all users
    user_num = len(user_model_update_list)
    root_update_vec = vectorize_net(root_update)
    trust_scores = torch.zeros(user_num)
    user_model_update_vecs = []
    for index, user_model_update in enumerate(user_model_update_list):
        user_model_update_vec = vectorize_net(user_model_update)
        user_model_update_vecs.append(user_model_update_vec)
        cos_sim = torch.cosine_similarity(user_model_update_vec, root_update_vec, dim=0)
        ts = torch.relu(cos_sim)
        trust_scores[index] = ts

    # 如果分数全为零，则返回上一次的全局模型
    if all(x == 0 for x in trust_scores):
        final_global_model_weights = copy.deepcopy(global_model_weights)
        return final_global_model_weights

    trust_scores_normalized = torch.div(trust_scores, trust_scores.sum())
    print(f"trust score: {trust_scores_normalized}")

    # # 规范化用户更新，通过与根更新对齐
    # norm_list = torch.zeros(user_num)
    # root_normal_number = torch.norm(root_update_vec)
    # for index, user_model_update_vec in enumerate(user_model_update_vecs):
    #     norm_list[index] = torch.div(root_normal_number, torch.norm(user_model_update_vec))
    #
    # for i, user_model_update in enumerate(user_model_update_list):
    #     for key, update_value in user_model_update.items():
    #         model_weights_list[i][key] = torch.mul(norm_list[i], update_value) + pre_global_model_weights[key]

    # 聚合：获取全局更新
    final_global_model_weights = copy.deepcopy(global_model_weights)
    for key in final_global_model_weights.keys():
        update = torch.zeros(final_global_model_weights[key].size()).to(device)
        for net_index, user_model in enumerate(model_weights_list):
            update += torch.mul(trust_scores_normalized[net_index], user_model[key])
        final_global_model_weights[key] = update

    return final_global_model_weights


def fltrust(model_weights_list, global_model_weights, root_train_loader, model_name, device):
    weight = fltrust_module(model_weights_list, global_model_weights, root_train_loader, model_name, device)
    runtime = fltrust_module.runtime
    malicious_score = [False for _ in range(len(model_weights_list))]
    return weight, runtime, malicious_score
