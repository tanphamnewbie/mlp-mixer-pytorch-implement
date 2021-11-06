from util import make_datapath_list
from lib import *
from config import *
from data_tranform import ImageTransform
from MLP_Mixer_model import MLPMixer


#Defire model & load weights

mlp_mixer = MLPMixer(image_shape=(256, 256), patch_size=patch_size, num_channels=3,
                     num_hidden_dim=128, num_layers=6, num_classes=num_classes, dropout=0.2, mlp_dim_factor=2)
print(mlp_mixer)
mlp_mixer.load_state_dict(torch.load('./exp/model_epoch_41.pth'))
mlp_mixer.eval()


#Preprocess test image
for filename in make_datapath_list(phase='test'):
    # Read and preprocess
    image = Image.open(filename)
    transform = ImageTransform(resize, mean, std)
    image = transform(image, phase="test")
    image = image.unsqueeze(0)
    image = image.to(device)

    # Visualize
    #img = torchvision.utils.make_grid(image)
    #img = img / 2 + 0.5  # Unnormalize
    #npimg = img.numpy()
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.show()

    # Run Inference
    output = torch.softmax(mlp_mixer(image).squeeze(), dim=0)
    print(output)
    output1 = output.detach().numpy()
    rst = np.argmax(output1)
    if rst == 0:
        print(filename[-6:] + ' Mồn lèo')
    else:
        print(filename[-6:] + ' Cờ hó')
