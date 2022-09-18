# In[6]:


import os
import io
import numpy as np


os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence

from PIL import Image
import nltk
#nltk.download('punkt')
import itertools

import time

import json

# In[7]:


file_list = os.listdir('Flickr8k_text')

# In[8]:


##Training images list
train_img_list = []
with open('Flickr8k_text/Flickr_8k.trainImages.txt', 'r') as f:
    for i in f:
        train_img_list.append(i.strip())

# In[9]:


##Test images list
test_img_list = []
with open('Flickr8k_text/Flickr_8k.testImages.txt', 'r') as f:
    for i in f:
        test_img_list.append(i.strip())


# In[10]:gi


img_caption = []
with open('Flickr8k_text/Flickr8k.token.txt', 'r') as f:
    for i in f:
        img_caption.append(i)

# In[11]:


##Store all the captions for each image
annot = {}
for i in range(0, len(img_caption), 5):
    ann = []
    t1 = img_caption[i].strip()
    for j in range(i, i + 5):
        tmp = img_caption[j].strip()
        tmp = tmp.split('\t')
        ann.append([tmp[1].lower()])
    t1 = t1.split('\t')
    annot[t1[0].split('#')[0]] = ann

# In[12]:


##Caption and Image List
cap_dict = {}
for i in range(0, len(img_caption), 5):
    tmp = img_caption[i].strip()
    tmp = tmp.split('\t')
    cap_dict[tmp[0].split('#')[0]] = tmp[1].lower()

# In[13]:


##Training captions
train_cap_dict = {}
for i in train_img_list:
    train_cap_dict[i] = cap_dict[i]

# In[14]:


##Test captions
test_cap_dict = {}
for i in test_img_list:
    test_cap_dict[i] = cap_dict[i]

# In[15]:


##Tokenize train captions
train_token = []
train_tok = []
for (j, i) in train_cap_dict.items():
    train_token.append([j, nltk.word_tokenize(i)])
    train_tok.append(nltk.word_tokenize(i))

# In[16]:


##Tokenize test captions
test_token = []
for (j, i) in test_cap_dict.items():
    test_token.append([j, nltk.word_tokenize(i)])

# In[17]:


##word_to_id and id_to_word
all_tokens = itertools.chain.from_iterable(train_tok)
word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}

all_tokens = itertools.chain.from_iterable(train_tok)
id_to_word = [token for idx, token in enumerate(set(all_tokens))]
id_to_word = np.asarray(id_to_word)

# In[18]:


##Sort the indices by word frequency

train_token_ids = [[word_to_id[token] for token in x[1]] for x in train_token]
count = np.zeros(id_to_word.shape)
for x in train_token_ids:
    for token in x:
        count[token] += 1
indices = np.argsort(-count)
id_to_word = id_to_word[indices]
count = count[indices]

# In[19]:


##Recreate word_to_id based on sorted list
word_to_id = {token: idx for idx, token in enumerate(id_to_word)}

# In[20]:


print("Vocabulary size: " + str(len(word_to_id)))

# In[16]:


## assign -4 if token doesn't appear in our dictionary
## add +4 to all token ids, we went to reserve id=0 for an unknown token
train_token_ids = [[word_to_id.get(token, -4) + 4 for token in x[1]] for x in train_token]
test_token_ids = [[word_to_id.get(token, -4) + 4 for token in x[1]] for x in test_token]

# In[17]:


word_to_id['<unknown>'] = -4
word_to_id['<start>'] = -3
word_to_id['<end>'] = -2
word_to_id['<pad>'] = -1

for (_, i) in word_to_id.items():
    i += 4
    word_to_id[_] = i

# In[18]:


id_to_word_dict = {}
cnt = 4
for i in id_to_word:
    id_to_word_dict[cnt] = i
    cnt += 1
id_to_word_dict[0] = '<unknown>'
id_to_word_dict[1] = '<start>'
id_to_word_dict[2] = '<end>'
id_to_word_dict[3] = '<pad>'

# In[19]:


##Length of each caption
train_cap_length = {}
for i in train_token:
    train_cap_length[i[0]] = len(i[1]) + 2

test_cap_length = {}
for i in test_token:
    test_cap_length[i[0]] = len(i[1]) + 2

# In[20]:


##Add <start> and <end> tokens to each caption
for i in train_token_ids:
    i.insert(0, word_to_id['<start>'])
    i.append(word_to_id['<end>'])

for i in test_token_ids:
    i.insert(0, word_to_id['<start>'])
    i.append(word_to_id['<end>'])

# In[21]:


##Pad train captions
length = []
for (i, j) in train_cap_length.items():
    length.append(j)
max_len = max(length)

for n, i in enumerate(train_token):
    if (train_cap_length[i[0]] < max_len):
        train_token_ids[n].extend(word_to_id['<pad>'] for i in range(train_cap_length[i[0]], max_len))

# In[22]:


##Convert token ids to dictionary for train
train_token_ids_dict = {}
for n, i in enumerate(train_token):
    train_token_ids_dict[i[0]] = train_token_ids[n]

# In[23]:


##Pad test captions
length = []
for (i, j) in test_cap_length.items():
    length.append(j)
max_len = max(length)

for n, i in enumerate(test_token):
    if (test_cap_length[i[0]] < max_len):
        test_token_ids[n].extend(word_to_id['<pad>'] for i in range(test_cap_length[i[0]], max_len))

# In[24]:


##Convert token ids to dictionary for test
test_token_ids_dict = {}
for n, i in enumerate(test_token):
    test_token_ids_dict[i[0]] = test_token_ids[n]

# In[25]:


## save dictionary
np.save('Flickr8k_text/flickr8k_dictionary.npy', np.asarray(id_to_word))

# In[26]:


## save training data to single text file
with io.open('Flickr8k_text/train_captions.txt', 'w', encoding='utf-8') as f:
    for i, tokens in enumerate(train_token_ids):
        f.write("%s " % train_token[i][0])
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")

# In[27]:


## save test data to single text file
with io.open('Flickr8k_text/test_captions.txt', 'w', encoding='utf-8') as f:
    for i, tokens in enumerate(test_token_ids):
        f.write("%s " % test_token[i][0])
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")


# ## Image preprocessing

# In[28]:


def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)


def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                #img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        if (i + 1) % 100 == 0:
            print("[{}/{}] Resized the images and saved into '{}'."
                  .format(i + 1, num_images, output_dir))


# In[39]:


##Resize image
image_dir = 'Flickr_Data/Flickr_Data/Images'
output_dir = 'Flickr8k_resized_image/'
image_size = [256, 256]
#resize_images(image_dir, output_dir, image_size)


# In[30]:


class Dataset(data.Dataset):
    def __init__(self, img_dir, img_id, cap_dictionary, cap_length, transform=None):
        self.img_dir = img_dir
        self.img_id = img_id
        self.transform = transform
        self.cap_dictionary = cap_dictionary
        self.cap_length = cap_length

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, index):
        img = self.img_id[index]
        img_open = Image.open(self.img_dir + img).convert('RGB')

        if self.transform is not None:
            img_open = self.transform(img_open)

        cap = np.array(self.cap_dictionary[img])
        cap_len = self.cap_length[img]

        return img_open, cap, cap_len


# In[31]:


transform_train = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

transform_test = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

# In[32]:


img_dir = 'Flickr8k_resized_image/'
train_data = Dataset(img_dir, train_img_list, train_token_ids_dict, train_cap_length, transform_train)

test_data = Dataset(img_dir, test_img_list, test_token_ids_dict, test_cap_length, transform_test)

# In[33]:


train_dataloader = data.DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = data.DataLoader(test_data, batch_size=32, shuffle=True)

# In[34]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# In[35]:


if torch.cuda.is_available():
    print('Yes')

# In[36]:

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-50 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


# In[37]:


encoder = EncoderCNN(1024)
decoder = DecoderRNN(1024, 1024, len(word_to_id), 1)

# In[38]:


for name, child in encoder.named_children():
    if name != 'linear' and name != 'bn':
        for name2, params in child.named_parameters():
            params.requires_grad = False


# In[39]:


##Function to sort the captions and images according to caption length
def sorting(image, caption, length):
    srt = length.sort(descending=True)
    image = image[srt[1]]
    caption = caption[srt[1]]
    length = srt[0]

    return image, caption, length


# In[40]:


#Loss and optimizer
encoder.to(device)
decoder.to(device)
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)

# In[41]:


##Train the model
# encoder.train()
# decoder.train()
# #
# train_loss = []
# time1 = time.time()
# epochs = 1
# total_step = len(train_dataloader)
# print("incepe train")
# for epoch in range(epochs):
#     print(epoch)
#     for i, (images, captions, lengths) in enumerate(train_dataloader):
#         print("a")
# #
#         images, captions, lengths = sorting(images, captions, lengths)
#         print("sorted")
# #
#         targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
#         print("a")
#         ##Forward,backward and optimization
#         features = encoder(images)
#         outputs = decoder(features, captions, lengths)
#         #outputs= torch.tensor(outputs, dtype=torch.long, device=device)
#         loss = criterion(outputs, targets.long())
#         decoder.zero_grad()
#         encoder.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         train_loss.append(loss)
# #
# #         # Print log info
#         if i % 100 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
#                   .format(epoch, epochs, i, total_step, loss.item(), np.exp(loss.item())))
# #
# #
# print('RUNNING TIME: {}'.format(time.time() - time1))
# #
# torch.save(encoder, os.path.join('models/flickr8k/', 'encoder.model'))
# torch.save(decoder, os.path.join('models/flickr8k/', 'decoder.model'))
class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X,captions,lengths, *args):
        enc_outputs = self.encoder(enc_X, *args)
        #dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(enc_outputs,captions,lengths)





model = EncoderDecoder(encoder, decoder)

#model=encoder
#model2=decoder

import onnx
from onnx import shape_inference

#from onnxsim import simplify

dummy_input =(torch.rand(1,3,500,375),torch.tensor([[   1,    4,   13,   10, 1130,   28,  165,    6,    7,   67,    8,  117,
            4,  313,   14,    7, 2955, 2396,    9,  138,    5,    2,    3,    3,
            3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
            3]], dtype=torch.int32),torch.tensor([22]))

torch.onnx.export(model, dummy_input, "showandtellenc.onnx", verbose=True, opset_version = 11)
model = onnx.load("showandtell.onnx")
inferred_model = shape_inference.infer_shapes(model)



# convert model
#model_simp, check = simplify(inferred_model)

#assert check, "Simplified ONNX model could not be validated"

#onnx.save(model, "my_model.onnx");