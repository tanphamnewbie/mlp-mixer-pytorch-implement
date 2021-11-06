from lib import *
from config import *
from util import make_datapath_list
from data_tranform import ImageTransform
from dataset import MyDataset
from MLP_Mixer_model import MLPMixer

#List image data
train_list = make_datapath_list("train")
print('Found {} image in  train dataset'.format(len(train_list)))
val_list = make_datapath_list("val")
print('Found {} image in  val dataset'.format(len(val_list)))

#Prepare data
train_dataset = MyDataset(train_list, transform=ImageTransform(resize, mean, std), phase="train")
val_dataset = MyDataset(val_list, transform=ImageTransform(resize, mean, std), phase="val")

#Load data
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)

#Difine model
mlp_mixer = MLPMixer(image_shape = (256, 256), patch_size = patch_size, num_channels = 3,
                     num_hidden_dim = 128, num_layers=6, num_classes=num_classes, dropout=0.2, mlp_dim_factor=2)
#Caculate parameters
parameters = filter(lambda p: p.requires_grad, mlp_mixer.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)

#Define optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_mixer.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

#Trainning
training_curve_data = {}

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for data, label in iter(train_dataloader):
        data = data.to(device)
        label = label.to(device)

        output = mlp_mixer(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_dataloader)
        epoch_loss += loss / len(train_dataloader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in val_dataloader:
            data = data.to(device)
            label = label.to(device)

            val_output = mlp_mixer(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(val_dataloader)
            epoch_val_loss += val_loss / len(val_dataloader)

    # ===============================================================================

    training_curve_data[epoch] = {'epoch_train_loss': float(epoch_loss.detach().numpy()),
                                  'epoch_val_loss': float(epoch_val_loss.detach().numpy()),
                                  'epoch_accuracy': float(epoch_accuracy.detach().numpy()),
                                  'epoch_val_accuracy': float(epoch_val_accuracy.numpy())}

    with open('training_curve_data.json', 'w') as fp:
        json.dump(training_curve_data, fp)

    if epoch % print_save_every == 0:
        print(
            f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
        torch.save(mlp_mixer.state_dict(), f'model_epoch_{epoch + 1}.pth')



