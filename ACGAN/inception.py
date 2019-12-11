import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import pdb
from utils import weights_init, compute_acc
from network import _netG, _netD, _netD_CIFAR10, _netG_CIFAR10

import torchvision.utils as vutils
from torchvision.models.inception import inception_v3
from embedders import BERTEncoder
import numpy as np
from scipy.stats import entropy

def inception_score(imgs, cuda=False, batch_size=32, resize=False, splits=10):
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

    nz = 200 #default size of latent vector for ACGAN
    ngpu = 0
    batch_size = 3200
    num_classes = 10
    embed_size = 100
    cifar_text_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    sentiment_text_labels = ['ancient city', 'broken glass', 'classic cars', 'cute dog', 'dead tree', 'falling leaves', 'hot pot', 'natural bridge', 'wild flowers', 'young lady']
    imagenet_text_labels = ['brown bear', 'shopping cart', 'seashore', 'crane', 'tree frog', 'carousel', 'frying pan', 'bookshop', 'basketball', 'cheeseburger']

    netG = _netG_CIFAR10(ngpu, nz)
    netG.apply(weights_init)
    # netG.load_state_dict(torch.load('output/netG_epoch_499.pth'))
    netG.load_state_dict(torch.load('ACGAN_sentiment_new.pth', map_location=torch.device('cpu')))

    netG.eval()

    eval_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)

    eval_noise = Variable(eval_noise)

    eval_noise_ = np.random.normal(0, 1, (batch_size, nz))
    eval_label = np.random.randint(0, num_classes, batch_size)

    encoder = BERTEncoder()
    
    captions = [imagenet_text_labels[per_label] for per_label in eval_label]
    print(captions)
    embedding = encoder(eval_label, captions)
    embedding = embedding.detach().numpy()

    eval_noise_[np.arange(batch_size), :embed_size] = embedding[:, :embed_size]
    eval_noise_ = (torch.from_numpy(eval_noise_))
    eval_noise.data.copy_(eval_noise_.view(batch_size, nz, 1, 1))


    fake = netG(eval_noise)
    print ("Calculating Inception Score...")
    print (inception_score(fake, cuda=False, batch_size=32, resize=True, splits=10))
    vutils.save_image(fake.data, "out.png")

    #print (inception_score(IgnoreLabelDataset(cifar), cuda=False, batch_size=32, resize=True, splits=10))
