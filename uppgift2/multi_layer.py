import torch
import torch.nn as nn
import torch.optim as optim

class MultiLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiLayer, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        out = self.hidden_layer(x)
        out = self.activation(out)
        out = self.output_layer(out)
        return out

# Skapa en instans av MLP-modellen
input_size = 10
hidden_size = 20
output_size = 1
model = MultiLayer(input_size, hidden_size, output_size)    

# Definiera träningsdata och måldata
input_data = torch.randn(100, input_size)
target_data = torch.randn(100, output_size)

# Definiera förlustfunktion och optimizer
criteria = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Träna modellen
epochs = 1000
for epoch in range(epochs):
    # Framåtpassagen
    predictions = model(input_data)
    loss = criteria(predictions, target_data)
    # Bakåtpassagen och uppdatera vikter
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Skriv ut förlusten varje 100:e epoch
    if epoch % 100 == 0:
        print(f"Epoch {epoch} Loss: {loss.item()}")

# Testa modellen
test_data = torch.randn(10, input_size)
predictions = model(test_data)
print("Predictions:", predictions)