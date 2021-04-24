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

EMBEDDING_FILE = {
    'glove': 'glove.pkl',
    'word2vec': 'word2vec.pkl'
}

IMAGE_DATA_SET = {
    'coco': 'coco.pkl',
    'news': 'news.pkl'
}

class ImageDataset(data.Dataset):
    def __init__(self, image_ids, captions, image_paths, vocab):
        self.image_ids = image_ids
        self.captions = captions
        self.image_paths = image_paths
        self.vocab = vocab
        self.transform = data_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.vocab_mean = np.zeros(50)
    
    # Return the vector for the token or the mean vector if the token is unknown   
    def _get_token_vec(self, token):
        return self.vocab[token] if token in self.vocab else self.vocab_mean
    
    # Return the image and caption at the index    
    def __getitem__(self, index):
        img_id = self.image_ids[index]
        caption = self.captions[index]
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to embedding vectors.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = [self._get_token_vec(token) for token in tokens]
        target = torch.Tensor(np.mean(caption, axis=0))
        return image, target

    def __len__(self):
        return len(self.image_ids)

# Create and return data loaders for train, validation, and test for the specified
# word embeddings and the specified image data set
def get_loaders(data_dir, img_data_set, embedding, batch_size, num_workers):
    
    # Open datasets
    
    data_set_path = os.path.join(data_dir, IMAGE_DATA_SET[img_data_set])
    datasets = pickle.load(open(data_set_path, 'rb'))
    # Open embedding
    embedding_path = os.path.join(data_dir, EMBEDDING_FILE[embedding])
    vocab = pickle.load(open(embedding_path, 'rb'))
    
    data_loaders = {}
    for ds, obj in datasets.items():
        image_ids = obj['image-ids']
        captions = obj['captions']
        image_paths = obj['image-paths']
        dataset = ImageDataset(image_ids, captions, image_paths, vocab)
        data_loaders[ds] = torch.utils.data.DataLoader(dataset=dataset, 
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)
    return data_loaders

# Create data files for the training, validation, and test sets for each image data set
def create_coco_image_sets(img_dir, img_data_file, output_dir):
    train_val_proportion = 0.08
    data_sets = defaultdict(list)
    coco = COCO(img_data_file)
    data_sets['train'] = list(coco.anns.keys())
    random.shuffle(data_sets['train'])
    train_val_size = math.floor(train_val_proportion * len(data_sets['train']))
    
    for i in range(train_val_size):
        data_sets['val'].append(data_sets['train'].pop())
        data_sets['test'].append(data_sets['train'].pop())
    
    coco_ds = {}
    for ds, ids in data_sets.items():
        coco_ds[ds] = {}
        coco_ds[ds]['image-ids'] = [coco.anns[i]['image_id'] for i in ids]
        coco_ds[ds]['captions'] = [coco.anns[i]['caption'] for i in ids]
        coco_ds[ds]['image-paths'] = [coco.loadImgs(img_id)[0]['file_name'] for img_id in coco_ds[ds]['image-ids']]
        coco_ds[ds]['image-paths'] = [os.path.join(img_dir, filename) for filename in coco_ds[ds]['image-paths']]
        
    output_file = os.path.join(output_dir, IMAGE_DATA_SET['coco'])
    print(f'Outputting COCO data sets as: {output_file}')
    pickle.dump(coco_ds, open(output_file, 'wb'))

# Normalize embeddings and use PCA to reduce dimensions to 50        
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

# Create and pickle GloVe and Word2Vec embeddinsg.  This includes normalizing using PCA to 
# reduce the dimensions to 50
def prepare_embeddings(embedding_file, output_dir):

    # Read in embeddings
    print('Reading GloVe embeddings...')
    words = []
    glove_embeddings = []
    
    with open(embedding_file, 'r') as f:
        for line in f:
            values = line.split()
            words.append(values[0])
            glove_embeddings.append(np.asarray(values[1:], "float32"))
       
    glove_embeddings = pd.DataFrame(glove_embeddings, index=words)
    glove_embeddings = normalize_reduce(glove_embeddings)
    
    output_file = os.path.join(args.output_dir, EMBEDDING_FILE['glove'])
    print(f'Outputting GloVe embeddings to file {output_file}')
    pickle.dump(glove_embeddings, open(output_file, 'wb'))
    
    # Reading w2v embeddings
    print('Reading Word2Vec embeddings...')
    w2v = gensim.downloader.load('word2vec-google-news-300')
    words = set(w2v.vocab).intersection(set(words))
    embeddings = [w2v[word] for word in words]
    w2v_embeddings = pd.DataFrame(embeddings, index=words)
    #w2v_embeddings = pd.DataFrame.from_dict(w2v_embeddings.wv)
    w2v_embeddings = normalize_reduce(w2v_embeddings)

    output_file = os.path.join(args.output_dir, EMBEDDING_FILE['word2vec'])
    print(f'Outputting Word2Vec embeddings to file {output_file}')
    pickle.dump(w2v_embeddings, open(output_file, 'wb')) 
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CS2770 HW3 Data Preparation')
    parser.add_argument('--output_dir', type=pathlib.Path, help='Output')
    parser.add_argument('--glove_embedding', type=pathlib.Path, help='The GloVe embedding file')
    parser.add_argument('--image_dir', type=pathlib.Path, help='Directory with image files')
    parser.add_argument('--image_data_file', type=pathlib.Path, help='JSON file with image data')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if args.glove_embedding is not None:
        prepare_embeddings(args.glove_embedding, args.output_dir)
   
    if args.image_dir is not None:
        create_coco_image_sets(args.image_dir, args.image_data_file, args.output_dir)   
