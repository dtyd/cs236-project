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
    print(num_batches)
    for (_, labels_batch, captions_batch) in dataloader:
        print(count)
        captions += captions_batch
        conditional_embeddings = encoder(labels_batch.to(device), captions)
        imgs = generative_model.sample(conditional_embeddings).cpu()
        gen_imgs.append(imgs)
    pdb.set_trace() #check for normalization
    gen_imgs = torch.cat(gen_imgs)
    gen_imgs = torch.load("file.pt")



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
    
    print ("Calculating Inception Score...")
    print (inception_score(gen_imgs, cuda=False, batch_size=batch_size, resize=True, splits=10)) # change or CUDA to true if necessary
    vutils.save_image(gen_imgs.data, "bert_epoch.png")

    #print (inception_score(IgnoreLabelDataset(cifar), cuda=False, batch_size=32, resize=True, splits=10))
