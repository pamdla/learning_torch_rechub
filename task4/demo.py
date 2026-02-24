import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch_rechub.models.multi_task import MMOE
from torch_rechub.trainers import MTLTrainer
from torch_rechub.utils.data import DataGenerator
from torch_rechub.basic.features import SparseFeature, DenseFeature

# 1. 定义特征
common_features = [
    SparseFeature(name="user_id", vocab_size=10000, embed_dim=32),
    SparseFeature(name="city", vocab_size=10, embed_dim=32),
    DenseFeature(name="age", embed_dim=1),
    DenseFeature(name="income", embed_dim=1),
]

# 2. 准备数据
if os.path.exists("sample_data.pkl"):
    os.remove("sample_data.pkl")
if not os.path.exists("sample_data.pkl"):
    n_samples = 10000
    user_ids = np.random.randint(0, 10000, n_samples)
    cities = np.random.randint(0, 10, n_samples)
    ages = np.random.uniform(18, 60, n_samples)
    incomes = np.random.uniform(1000, 100000, n_samples)

    X = {
        "user_id": user_ids,
        "city": cities,
        "age": ages,
        "income": incomes,
    }

    # Task1: CTR 受age和income影响
    logit1 = -2 + 0.05 * (ages - 30) + 0.00003 * (incomes - 30000)
    prob1 = 1 / (1 + np.exp(-logit1))
    y_task1 = (np.random.random(n_samples) < prob1).astype(int)

    # Task2: CVR 受city和age影响 (增强信号)
    logit2 = -2 + 0.5 * (cities - 5) + 0.15 * (ages - 40)
    prob2 = 1 / (1 + np.exp(-logit2))
    y_task2 = (np.random.random(n_samples) < prob2).astype(int)

    y = np.column_stack([y_task1, y_task2])
    joblib.dump([X, y], "sample_data.pkl")
else:
    X, y = joblib.load("sample_data.pkl")
    X = {k: np.array(v) for k, v in X.items()}

# 3. 创建数据生成器
dg = DataGenerator(X, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    split_ratio=[0.7, 0.1], batch_size=128
)

# 4. 创建模型
model = MMOE(
    features=common_features,
    task_types=["classification", "classification"],
    n_expert=16,
    expert_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [2 * 64, 2 * 32], "dropout": 0.2, "activation": "relu"},
        {"dims": [2 * 64, 2 * 32], "dropout": 0.2, "activation": "relu"},
    ],
)

# 5. 创建训练器
trainer = MTLTrainer(
    model=model,
    task_types=["classification", "classification"],
    optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
    adaptive_params={"method": "uwl"},
    n_epoch=200,
    earlystop_patience=10,
    device="cpu",
)

# 6. 训练模型
trainer.fit(train_dl, val_dl)

# 7. 评估模型
scores = trainer.evaluate(trainer.model, test_dl)
print(f"Task 1 AUC: {scores[0]}")
print(f"Task 2 AUC: {scores[1]}")

# 8. 导出ONNX模型
trainer.export_onnx("mmoe.onnx")

# 9. 模型预测
model = trainer.model
model.eval()
n_samples = len(test_dl.dataset)
n_tasks = 2
preds = np.zeros((n_samples, n_tasks))
start_idx = 0
with torch.no_grad():
    for x, y in test_dl:
        x = {k: v.to(trainer.device) for k, v in x.items()}
        output = model(x)
        batch_size = output.shape[0]
        preds[start_idx : start_idx + batch_size] = output.cpu().numpy()
        start_idx += batch_size
print(f"Predictions shape: {preds.shape}")

# 保存预测结果为CSV
preds_df = pd.DataFrame(preds, columns=["task1_pred", "task2_pred"])
preds_df.to_csv("predictions.csv", index=False)
print(f"Predictions saved to predictions.csv")
