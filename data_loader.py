import os, math, nltk, random, requests, numpy as np, torch
import torchvision.transforms as transforms
import torch.utils.data as data
from collections import defaultdict
from PIL import Image
from pycocotools.coco import COCO

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, coco, ids, vocab, transform=None):
        self.root = root
        self.coco = coco
        self.ids = ids
        self.vocab = vocab
        self.transform = transform
        print(f'Size of vocab: {self.vocab.size()}')
        self.vocab_mean = np.mean(self.vocab, axis=0)
        print(f'Mean size: {self.vocab_mean.size()}')

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
        caption = [self._get_token(token) for token in tokens]
        target = torch.Tensor(np.mean(caption, axis=0))
        return image, target

    def __len__(self):
        return len(self.ids)

def get_loaders(root, json_file, embedding_file, transform, batch_size, shuffle, num_workers):

    # Divide the data set up in training, validation, and test sets
    train_val_proportion = 0.08
    data_sets = defaultdict(list)
    coco = COCO(json_file)
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
        coco_ds =  CocoDataset(root, coco, ids, vocab, transform)
        coco_ds.__getitem__(5)
        data_loaders[ds] = torch.utils.data.DataLoader(dataset=coco_ds, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loaders
