import argparse, gensim, gensim.downloader
import os, math, nltk
import pandas as pd, pathlib, pickle
import random, requests, numpy as np, torch
import torchvision.transforms as transforms
import torch.utils.data as data
from collections import defaultdict
from PIL import Image
from pycocotools.coco import COCO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

GLOVE_OUTPUT_FILE = 'glove.pkl'
WORD2VEC_OUTPUT_FILE = 'word2vec.pkl'

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, coco, ids, vocab):
        self.root = root
        self.coco = coco
        self.ids = ids
        self.vocab = vocab
        self.transform = data_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.vocab_mean = np.mean(np.stack(list(self.vocab.values()), 0), axis=0)

    def _get_token_vec(self, token):
        return self.vocab[token] if token in self.vocab else self.vocab_mean
        
    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        img_path = os.path.join(self.root, coco.loadImgs(img_id)[0]['file_name'])
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to embedding vectors.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = [self._get_token_vec(token) for token in tokens]
        target = torch.Tensor(np.mean(caption, axis=0))
        return image, target

    def __len__(self):
        return len(self.ids)

# Create and return data loaders for train, validation, and test for the specified
# word embeddings and the specified image data set
def get_loaders(root, img_data_file, embedding_file, batch_size, num_workers):

    # Divide the data set up in training, validation, and test sets
    train_val_proportion = 0.08
    data_sets = defaultdict(list)
    coco = COCO(img_data_file)
    data_sets['train'] = list(coco.anns.keys())
    random.shuffle(data_sets['train'])
    train_val_size = math.floor(train_val_proportion * len(data_sets['train']))
    
    for i in range(train_val_size):
        data_sets['val'].append(data_sets['train'].pop())
        data_sets['test'].append(data_sets['train'].pop())
    
    # Create an embedding dictionary
    vocab = {}
    with open(embedding_file, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            vocab[word] = vector
    
    data_loaders = {}
    for ds, ids in data_sets.items():
        coco_ds =  CocoDataset(root, coco, ids, vocab)
        coco_ds.__getitem__(5)
        data_loaders[ds] = torch.utils.data.DataLoader(dataset=coco_ds, 
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)
    return data_loaders

def normalize_reduce(embeddings):
    # Normalize the embeddings
    scaler = StandardScaler()
    scaler.fit(embeddings)
    embeddings = scaler.transform(embeddings)

    # Reduce dimensions using PCA
    pca = PCA(n_components=50)
    pca.fit(embeddings)
    embeddings = pca.transform(embeddings)
    
    return embeddings

# Create and pickle the embedding.  This includes normalizing and, if necessary, using PCA to 
# reduce the dimensions to 50
def prepare_embeddings(embedding_file, output_dir):

    # Read in embeddings
    words = []
    glove_embeddings = []
    
    with open(embedding_file, 'r') as f:
        for line in f:
            values = line.split()
            words.append(values[0])
            glove_embeddings.append(np.asarray(values[1:], "float32"))
       
    glove_embeddings = pd.DataFrame(glove_embeddings, index=words)
    glove_embeddings = normalize_reduce(glove_embeddings)
    
    w2v = gensim.downloader.load('word2vec-google-news-300')
    
    word2vec_embeddings = [w2v.wv[word] for word in words]
    word2vec_embeddings = pd.DataFrame(word2vec_embeddings, index=words)
    word2vec_embeddings = normalize_reduce(word2vec_embeddings)
    
    # Output
    output_file = os.path.join(args.output_dir, GLOVE_OUTPUT_FILE) 
    pickle.dump(glove_embeddings, open(output_file, 'wb'))

    output_file = os.path.join(args.output_dir, WORD2VEC_OUTPUT_FILE) 
    pickle.dump(word2vec_embeddings, open(output_file, 'wb'))    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CS2770 HW3 Data Preparation')
    parser.add_argument('--output_dir', type=pathlib.Path, help='Output')
    parser.add_argument('--glove_embedding', type=pathlib.Path, help='The GloVe embedding file')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    prepare_embeddings(args.glove_embedding, args.output_dir)
   
    
