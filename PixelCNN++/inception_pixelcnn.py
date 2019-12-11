import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from models.embedders import BERTEncoder, OneHotClassEmbedding, UnconditionalClassEmbedding
from models.pixelcnnpp import ConditionalPixelCNNpp
from data import CIFAR10Dataset, Imagenet32Dataset
import pdb


import torchvision.utils as vutils
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy

def inception_score(imgs, cuda=False, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    #pdb.set_trace()
    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        print(i)
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':


    #NEW STUFF PixelCNN

    pdb.set_trace()

    n_resnet = 5
    n_filters = 160

    total = 2 #number of images to generate
    batch_size = 2 #number of images to generate at atime

    val_dataset = Imagenet32Dataset(train=0, max_size = total)

    dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    encoder = BERTEncoder()
    generative_model = ConditionalPixelCNNpp(embd_size=encoder.embed_size, img_shape=val_dataset.image_shape,
                                             nr_resnet=n_resnet, nr_filters = n_filters,
                                             nr_logistic_mix=3 if val_dataset.image_shape[0] == 1 else 10)


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    generative_model = generative_model.to(device)
    encoder = encoder.to(device)

    generative_model.load_state_dict(torch.load('checkpoints/bert_epoch_2.pt', map_location=torch.device('cpu'))) #replace with model and device

    generative_model.eval()

    captions = []
    gen_imgs = []
    done = False

    num_batches = int(total/batch_size)
    count = 0
    # print(num_batches)
    # for (_, labels_batch, captions_batch) in dataloader:
    #     print(count)
    #     captions += captions_batch
    #     conditional_embeddings = encoder(labels_batch.to(device), captions)
    #     imgs = generative_model.sample(conditional_embeddings).cpu()
    #     gen_imgs.append(imgs)
    # pdb.set_trace() #check for normalization
    # gen_imgs = torch.cat(gen_imgs)
    gen_imgs = torch.load("file.pt")


    #normalize images of size[total, 3, 32, 32] here, looks like this

    # tensor([[[[0.9647, 0.9565, 0.9808,  ..., 0.7953, 0.6243, 0.5831],
    #       [0.9612, 0.9186, 0.9765,  ..., 0.7327, 0.6981, 0.6150],
    #       [0.9540, 0.8860, 0.9710,  ..., 0.7219, 0.7349, 0.7974],
    #       ...,
    #       [1.0000, 0.7405, 0.5251,  ..., 0.6545, 0.6328, 0.7679],
    #       [1.0000, 0.7683, 0.6572,  ..., 0.7564, 0.6740, 0.6442],
    #       [0.9371, 0.7441, 0.6325,  ..., 0.6097, 0.5758, 0.4886]],

    #      [[0.9663, 0.9594, 0.9823,  ..., 0.7243, 0.4936, 0.4763],
    #       [0.9647, 0.9147, 0.9825,  ..., 0.6360, 0.5933, 0.4783],
    #       [0.9562, 0.8730, 0.9776,  ..., 0.4536, 0.3995, 0.5894],
    #       ...,
    #       [0.9511, 0.7521, 0.5613,  ..., 0.5191, 0.5335, 0.7019],
    #       [0.9607, 0.7699, 0.6847,  ..., 0.6680, 0.5805, 0.5553],
    #       [0.9257, 0.8018, 0.6732,  ..., 0.5472, 0.5357, 0.4214]],

    #      [[0.9701, 0.9655, 0.9837,  ..., 0.6608, 0.3901, 0.3730],
    #       [0.9697, 0.9276, 0.9862,  ..., 0.5760, 0.5360, 0.3101],
    #       [0.9606, 0.8872, 0.9818,  ..., 0.3843, 0.3044, 0.5081],
    #       ...,
    #       [0.8654, 0.9185, 0.8064,  ..., 0.7289, 0.6377, 0.8190],
    #       [0.8812, 0.9371, 0.9113,  ..., 0.8472, 0.7012, 0.6523],
    #       [0.9411, 0.9974, 0.9283,  ..., 0.6375, 0.6073, 0.4986]]],


    #     [[[0.4813, 0.4737, 0.4717,  ..., 0.6296, 0.5739, 0.2882],
    #       [0.5172, 0.5212, 0.5247,  ..., 0.6077, 0.5225, 0.6223],
    #       [0.5458, 0.5474, 0.5527,  ..., 0.4984, 0.3298, 0.6590],
    #       ...,
    #       [0.6171, 0.5835, 0.5408,  ..., 0.4292, 0.3384, 0.0188],
    #       [0.4328, 0.5094, 0.5758,  ..., 0.4320, 0.1080, 0.0000],
    #       [0.0347, 0.0523, 0.1858,  ..., 0.1896, 0.0000, 0.0000]],

    #      [[0.4297, 0.4253, 0.4204,  ..., 0.3897, 0.3865, 0.2237],
    #       [0.4578, 0.4687, 0.4771,  ..., 0.2282, 0.3241, 0.5158],
    #       [0.4893, 0.4942, 0.5008,  ..., 0.1531, 0.1835, 0.5351],
    #       ...,
    #       [0.7799, 0.7473, 0.6920,  ..., 0.2115, 0.1434, 0.0000],
    #       [0.4260, 0.5873, 0.7302,  ..., 0.2132, 0.0721, 0.0241],
    #       [0.0000, 0.0344, 0.1644,  ..., 0.0870, 0.0457, 0.0405]],

    #      [[0.2544, 0.2516, 0.2595,  ..., 0.3315, 0.3173, 0.1365],
    #       [0.2853, 0.2929, 0.3233,  ..., 0.2665, 0.2394, 0.4318],
    #       [0.3258, 0.3298, 0.3482,  ..., 0.1972, 0.1202, 0.4393],
    #       ...,
    #       [1.0000, 0.9812, 0.9024,  ..., 0.3393, 0.2574, 0.1071],
    #       [0.6637, 0.8170, 0.9591,  ..., 0.3583, 0.1993, 0.1343],
    #       [0.1212, 0.1653, 0.3459,  ..., 0.2594, 0.2213, 0.1800]]]])


    # normalize across each channels for each sample
    x = torch.reshape(gen_imgs, (gen_imgs.size(0), gen_imgs.size(1), gen_imgs.size(2)*gen_imgs.size(3))) # reshaped to (total, 3, 32*32)
    min_x = torch.min(x, 2)
    max_x = torch.max(x, 2) # tuple, with element 0 being the mins, element 1 being argmins (size (total, 3) at each element)

    # apply transform over each channel for each example 
    x_renorm = torch.zeros_like(gen_imgs)

    # renormalize to [-1, 1] 
    for sample in range(x_renorm.size(0)):
        for channel in range(x_renorm.size(1)):
            #print(x_renorm[sample][channel].size())
            x_renorm[sample][channel] = 2*(gen_imgs[sample][channel] - min_x[0][sample][channel].repeat(32,32)) / (max_x[0][sample][channel].repeat(32,32) - min_x[0][sample][channel].repeat(32,32)) - 1

    gen_imgs = x_renorm
    # feed x_renorm to inception score

    print (inception_score(gen_imgs, cuda=False, batch_size=batch_size, resize=True, splits=10)) # change or CUDA to true if necessary
    vutils.save_image(gen_imgs.data, "bert_epoch.png")

    print ("Calculating Inception Score...")
    #print (inception_score(IgnoreLabelDataset(cifar), cuda=False, batch_size=32, resize=True, splits=10))
