import os, math, nltk, random, numpy as np, torch
import torchvision.transforms as transforms
import torch.utils.data as data
from collections import defaultdict
from PIL import Image
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, ids, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = ids
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        print(f'Caption is: {caption}')
        # Convert caption (string) to embedding vectors.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = [vocab(token) for token in tokens]
        target = torch.Tensor(np.mean(caption, axis=1))
        print(f'Target is {target}')
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loaders(root, json_file, embedding_file, transform, batch_size, shuffle, num_workers):

    # Divide the data set up in training, validation, and test sets
    train_val_proportion = 0.08
    data_sets = defaultdict(list)
    coco = COCO(json)
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
        coco =  CocoDataset(root, ids, vocab, transform)
        coco.__getitem__(5)
        data_loaders[ds] = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    

    return data_loaders
