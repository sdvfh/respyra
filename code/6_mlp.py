import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from utils import created_targets, path


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return torch.softmax(x, dim=1)


device = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_parquet(path["data"] / "full_with_suggestions_and_scores.snappy.parquet")

df = df[df["disease"] != "control"]
completion_variables = [f"score_completion_{target}" for target in created_targets]
sentence_variables = [f"score_sentence_{target}" for target in created_targets]

X = df[completion_variables + sentence_variables + ["type"]]
y = df["disease"]

output_size = len(y.unique())
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train = X[X["type"] == "train"].drop(columns=["type"])
y_train = y[X["type"] == "train"]
X_val = X[X["type"] == "valid"].drop(columns=["type"])
y_val = y[X["type"] == "valid"]
X_test = X[X["type"] == "test"].drop(columns=["type"])
y_test = y[X["type"] == "test"]
input_size = X_train.shape[1]

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = SimpleNeuralNetwork(input_size, output_size)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

num_epochs = 100000
best_val_loss = float("inf")
patience = 5
wait_count = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            val_outputs = model(inputs)
            val_loss = criterion(val_outputs, labels)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        wait_count = 0
    else:
        wait_count += 1
        if wait_count >= patience:
            print(
                f"Early stopping at epoch {epoch + 1} as validation loss didn't improve."
            )
            break

predictions = model(X_test_tensor).detach().cpu().numpy()
y_true = y_test
y_pred = predictions
y_pred_round = predictions.argmax(axis=1)

diseases_accuracy = jaccard_score(y_true, y_pred_round, average=None)
diseases_accuracy = dict(zip(label_encoder.classes_, diseases_accuracy, strict=True))
