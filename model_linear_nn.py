import preprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import f1_score
import torch.nn.init as init


class LinearNN(nn.Module):
    def __init__(self, input_size):
        super(LinearNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 1)
        self.relu = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)  # add a dropout layer with dropout probability of 0.2

        # Xavier (Glorot) Initialization
        init.xavier_uniform_(self.layer1.weight)
        init.xavier_uniform_(self.layer2.weight)
        init.xavier_uniform_(self.layer3.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)  # add dropout after the first layer
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)  # add dropout after the second layer
        x = self.layer3(x)
        return x


if __name__ == '__main__':
    path_to_train = 'data/train/'
    path_to_test = 'data/test/'
    train = preprocess.load_data_to_dict(path_to_train)
    test = preprocess.load_data_to_dict(path_to_test)
    train = preprocess.preprocess(train)
    test = preprocess.preprocess(test)

    # Define hyperparameters
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 32

    # Convert train data to PyTorch tensors and create data loader
    train_x = torch.tensor(train.drop(columns=['SepsisLabel']).values, dtype=torch.float32)
    train_y = torch.tensor(train['SepsisLabel'].values, dtype=torch.float32)
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_x = torch.tensor(test.drop(columns=['SepsisLabel']).values, dtype=torch.float32)

    # Define the model
    input_size = len(train.columns) - 1
    model = LinearNN(input_size)

    # Define the weights for each class
    pos_weight = torch.tensor(10)  # Give more emphasis to label 1 by setting its weight to 10

    # Define the criterion with the pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    results = []
    for epoch in range(num_epochs):
        model.train()
        accuracy = 0
        loss_val = 0
        f1 = 0
        for x_batch, y_batch in train_loader:
            # Forward pass
            out = model(x_batch)
            # Compute loss

            loss = criterion(out.squeeze(), y_batch)

            y_pred = out.clone().detach().numpy()
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0
            y_pred = torch.tensor(y_pred).view(y_batch.shape)
            accuracy += (y_pred == y_batch).sum()
            f1 += f1_score(y_batch, y_pred, zero_division=1) * x_batch.size(0)
            loss_val += loss.item() * x_batch.size(0)
            # Backward pass and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss_val / len(train_loader.dataset):.4f}, Train Accuracy: {accuracy / len(train_loader.dataset):.4f}, Train f1: {f1 / len(train_loader.dataset):.4f}')

        # Evaluate the model on test data
        with torch.no_grad():
            model.eval()
            y_pred = model(test_x)
            y_pred = y_pred.squeeze().numpy()
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0

        # Calculate accuracy
        test_y = test['SepsisLabel'].values
        accuracy = (y_pred == test_y).mean()
        f1 = f1_score(test_y, y_pred)
        results.append(f1)
        print(f'Test Accuracy: {accuracy:.4f}, F1: {f1:.4f}')
    print(max(results))
