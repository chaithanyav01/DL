# Linear Regression using PyTorch

#import necessary libraries
import torch
import torch.nn as nn
from tqdm import tqdm

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate synthetic data
x_train = torch.tensor([x for x in range(10)])
y_train = torch.tensor([2 * x + 5 for x in range(10)])

# Define the model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 50)
        self.linear2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
# Initialize model, loss function, and optimizer
model = LinearRegressionModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1000

# Training loop
def train():
    #track progress in epochs
    pbar = tqdm(total=num_epochs, desc="Training", unit="step")

    for epoch in range(num_epochs):
        model.train()

        for i in range(len(x_train)):
            inputs = x_train[i].view(1, -1).float().to(device)
            targets = y_train[i].view(1, -1).float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            
        pbar.set_postfix(epoch=epoch+1, loss=loss.item())
        pbar.update(1)

    pbar.close()

train()

# Save the model
torch.save(model.state_dict(), 'linear_regression_model.pth')
print("Model saved to linear_regression_model.pth")

# Load the model
model_loaded = LinearRegressionModel().to(device)
model_loaded.load_state_dict(torch.load('linear_regression_model.pth'))

#infernce 
def infernce(input_value):
    model_loaded.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([[input_value]]).float().to(device)
        predicted = model_loaded(input_tensor)
        return predicted.item()

# Example inference
if __name__ == "__main__":
    test_value = 15
    prediction = infernce(test_value)
    print(f'Prediction for input {test_value}: {prediction}')