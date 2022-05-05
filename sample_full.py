import torch
import torch.nn as nn
from torch import optim
import pickle
import random
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
time_s=10
points_depth=3
points_hand=21
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        global time_s
        global points_depth
        global points_hand
        self.encoder_hidden_layer = nn.Linear(
            in_features=points_hand*points_depth*time_s, out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=64
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=64, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=points_depth*points_hand*time_s
        )

    def forward(self, x):
        activation = self.encoder_hidden_layer(x)
        activation = torch.selu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.selu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.selu(activation)
        reconstructed = self.decoder_output_layer(activation)
        return reconstructed

class Data():
    def __init__(self, batch):
        self.batchsize = batch
        self.samples = []
        file = open("0000000.obj", 'rb')
        List = pickle.load(file)
        file = open("0000001.obj", 'rb')
        List += pickle.load(file)
        file = open("0000002.obj", 'rb')
        List += pickle.load(file)
        file = open("0000003.obj", 'rb')
        List += pickle.load(file)
        file = open("0000004.obj", 'rb')
        List += pickle.load(file)
        tmp=[]
        for j, obj in enumerate(List):
            if len(obj)>0:
                tmp.append(obj)
                if len(tmp)>10:
                    tmp.pop(0)
                    tmp2 = np.asarray(tmp)
                    tmp2[:,:,2]+=0.5
                    self.samples.append(tmp2)
            else:
                tmp.clear()
        print(len(self.samples))

    def __len__(self):
        return len(self.samples) % self.batchsize

    def __getitem__(self, idx):
        global time_s
        global points_depth
        global points_hand
        All_data = []
        All_labels = []
        for b in range(self.batchsize):
            obj = random.randrange(len(self.samples))
            data = self.samples[obj].copy()
            All_labels.append(data.copy())
            draw_all=False
            for j in range(1,10):
                r = random.random()
                if r<0.15 or draw_all:
                    draw_all=True
                    for i in range(len(data[j])):
                        data[j][i] = (0.5,0.5,0.5)
                if r>0.15:
                    draw_all=False
            r = random.random()
            if r < 0.90:
                for i in range(len(data[j])):
                    data[9][i] = (0.5, 0.5,0.5)
            All_data.append(data.copy())


        All_data = torch.FloatTensor(All_data)
        All_data = All_data.reshape(-1, time_s*points_hand*points_depth)
        All_labels =torch.FloatTensor(All_labels)
        All_labels = All_labels.reshape(-1, time_s*points_hand*points_depth)
        return All_data, All_labels

model = Net()
print(model)

random_data = torch.rand((2, time_s*points_hand*points_depth))

result = model(random_data)
print (result)


model.to(device)
train_loader = Data(batch=64)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

num_epochs=200
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i in range(total_step):
        data, labels = train_loader.__getitem__(i)
        # Move tensors to the configured device
        data = data.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)

        # Backprpagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

torch.save(model.state_dict(), 'model3.ckpt')