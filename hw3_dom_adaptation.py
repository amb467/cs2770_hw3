import argparse, copy, os, pathlib, random, torch
import torch.nn as nn, torch.optim as optim
import torchvision.models as models
from torch.optim import lr_scheduler
from torchvision import transforms
from data_loader import get_loaders
from hw3 import dim_reduce

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def derange(l):
  if len(l) == 1:
    return None
  l_copy = l.copy()
  random.shuffle(l_copy)
  new_l = []

  for i in range(len(l)):
    if l[i] == l_copy[0]:
      l_copy = derange(l_copy)
      if l_copy is None:
        return derange(l)
    new_l.append(l_copy.pop(0))
  return new_l

def make_derangement(ts):
  l = ts.tolist()
  l = derange(l)
  return torch.Tensor(l)
   
def train(epochs, model, model_path, coco_data_loaders, news_data_loaders):

    model.to(device)
    linear = nn.Linear(1000, 2)
    linear.to(device)

    retrieval_criterion = nn.TripletMarginLoss()
    domain_predict_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = None
    best_acc = 0.0
    
    # Train the models
    for epoch in range(1,epochs+1):
        print(f'Epoch {epoch} of {epochs}')
    
        model.train()
        
        for i, (inputs, targets) in enumerate(coco_data_loaders['train']):
            print(f"Training source batch {i} of {len(coco_data_loaders['train'])}")
        
            # Set mini-batch dataset
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate the loss for retrieval
            negatives = make_derangement(targets)
            negatives.to(device)
            retrieval_loss = retrieval_criterion(dim_reduce(outputs), targets, negatives)
            
            # Calculate the loss from domain prediction
            domain_predict_loss = domain_predict_criterion(linear(outputs), torch.Tensor([1]))
            
            # Combine losses
            loss = torch.sub(retrieval_loss, domain_predict_loss)
            loss.backward()
            optimizer.step()
        
        target_len = math.floor(len(news_data_loaders['train']) / 10)      
        for i, (inputs, targets) in enumerate(coco_data_loaders['train']):
            if i > target_len:
                break
        
            # Set mini-batch dataset
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
                        
            # We only use domain prediction loss here
            loss = -1 * domain_predict_criterion(linear(outputs), torch.Tensor([0]))
            loss.backward()
            optimizer.step()        
    
        scheduler.step()
        
        image_to_text, text_to_image = get_test_results(model, coco_data_loaders['val'], news_data_loaders['val'])
        if image_to_text > best_acc:
            best_acc = image_to_text
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, model_path)
        
def get_test_results(model, coco_data_loader, news_data_loader):

    model.eval()
    image_to_text = []
    text_to_image = []
    
    for i, (inputs, targets) in enumerate(coco_data_loader):
        print(f"Validating COCO batch {i} of {len(data_loader)}")
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = dim_reduce(model(inputs))
        distances = torch.cdist(targets, outputs)
        
        values, indices = torch.min(distances, 0)
        for n, i in enumerate(indices):
            image_to_text.append(1.0 if n == int(i) else 0.0)

        values, indices = torch.min(distances, 1)
        for n, i in enumerate(indices):
            text_to_image.append(1.0 if n == int(i) else 0.0)

    for i, (inputs, targets) in enumerate(news_data_loader):   
        print(f"Validating Good News batch {i} of {len(data_loader)}")
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = dim_reduce(model(inputs))
        distances = torch.cdist(targets, outputs)
        
        values, indices = torch.min(distances, 0)
        for n, i in enumerate(indices):
            image_to_text.append(1.0 if n == int(i) else 0.0)

        values, indices = torch.min(distances, 1)
        for n, i in enumerate(indices):
            text_to_image.append(1.0 if n == int(i) else 0.0)   
        
    image_to_text = sum(image_to_text) / float(len(image_to_text))
    text_to_image = sum(text_to_image) / float(len(text_to_image))
    return image_to_text, text_to_image
    
if __name__ == "__main__":

    # Get arguments
    parser = argparse.ArgumentParser(description='CS2770 HW3')
    parser.add_argument('--epochs', type=int, default=10, help='The number of epochs')
    parser.add_argument('--data_dir', type=pathlib.Path, help='The directory where image and embedding pickle files can be found')
    parser.add_argument('--output_dir', type=pathlib.Path, help='Output')
    args = parser.parse_args()
              
    if not os.path.exists(args.data_dir):
        raise Exception('Not a valid path to find image and embedding pickle files: {args.data_dir}')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create parameters and model
    model_path = os.path.join(args.output_dir, f'best_cross_domain_adaptation.pth')
    batch_size = 128
    num_workers = 2
    model = models.alexnet(pretrained=True)
    
    # Training
    print(f'Cross-domain adaptation: training')
    coco_data_loaders = get_loaders(args.data_dir, "coco", "glove", batch_size, num_workers)
    news_data_loaders = get_loaders(args.data_dir, "news", "glove", batch_size, num_workers)
    train(args.epochs, model, model_path, coco_data_loaders, news_data_loaders)
    model.load_state_dict(torch.load(model_path))
    
    # Testing COCO
    print(f'Testing COCO')
    image_to_text, text_to_image = get_test_results(model, coco_data_loaders['test'])
    print(f'COCO Model accuracy: image-to-text {image_to_text}; text-to-image {text_to_image}')
    
    # Testing Good News
    print(f'Testing Good News')
    image_to_text, text_to_image = get_test_results(model, news_data_loaders['test'])
    print(f'Good News Model accuracy: image-to-text {image_to_text}; text-to-image {text_to_image}')    