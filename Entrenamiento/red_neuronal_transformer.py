import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np

# Datos
X = torch.tensor([
    [255, 0, 0], [200, 20, 10], [150, 50, 40],
    [0, 255, 0], [10, 200, 15], [5, 150, 5],
    [0, 0, 255], [15, 10, 200], [5, 5, 150]
], dtype=torch.float32) / 255

Y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)

EPOCHS = 1000

class TransformerModel(nn.Module):
    def __init__(self, dim_entradas=3, num_clases=3, d_model=32, nhead=4, num_capas=2):
        super(TransformerModel, self).__init__()
        
        # Proyectamos los 3 valores RGB a d_model (dimensión interna del Transformer)
        self.embedding = nn.Linear(dim_entradas, d_model)
        
        # Capa Transformer (Encoder)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_capas)
        
        # Capa final de clasificación
        self.classifier = nn.Linear(d_model, num_clases)

    def forward(self, x):
        # x shape: [batch_size, 3] -> lo convertimos a [batch_size, 1, 3] (secuencia de longitud 1)
        x = x.unsqueeze(1)
        
        # Proyección y Transformer
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        
        # Tomamos la salida y clasificamos (quitamos la dimensión de secuencia)
        x = x.mean(dim=1) 
        return self.classifier(x)
    
model = TransformerModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    test_color = torch.tensor([[255, 0, 200]], dtype=torch.float32) / 255.0
    prediction = model(test_color)
    predicted_class = torch.argmax(prediction, dim=1).item()
    colors = ["Rojo", "Verde", "Azul"]
    print(f"El color predicho es: {colors[predicted_class]}")