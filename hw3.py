import argparse, copy, os, pathlib, random, torch
import torch.optim as optim
import torchvision.models as models
from torch.nn import AvgPool1d, TripletMarginLoss
from torch.optim import lr_scheduler
from torchvision import transforms
from data_loader import get_loaders
from torchsummary import summary

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# t is a tensor with dimensions N x 1000, return a tensor with dimensions N x 50
def dim_reduce(t):
    m = AvgPool1d(20)   
    t = m(t.unsqueeze(1))
    return t.squeeze()
 
# Make a complete derangement of tensor t
def make_derangement(t):

    t_list = list(t) 
    new_indices = []
    indices = [0]

    while len(new_indices) < len(t_list):

        if len(indices) == 1 and indices[0] == len(new_indices):
            indices = list(range(len(t_list)))
            new_indices = []
            continue
        
        random.shuffle(indices)
    
        while len(indices) > 0 and indices[0] != len(new_indices):
            new_indices.append(indices.pop(0))

    new_list = [t_list[i] for i in new_indices]
    return torch.stack(new_list, 0)
   
def train(epochs, model, data_loaders, model_path):

    model.to(device)
    #summary(model, (3, 299, 299))

    criterion = TripletMarginLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = None
    best_acc = 0.0
    
    # Train the models
    for epoch in range(1,epochs+1):
        print(f'Epoch {epoch} of {epochs}')
    
        model.train()
        
        for inputs, targets in data_loaders['train']:
            # Set mini-batch dataset
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = dim_reduce(model(inputs))
    
            # Make a derangement of targets so the negative is always different from positive
            negatives = make_derangement(targets)
            loss = criterion(outputs, targets, negatives)
            loss.backward()
            optimizer.step()
    
        scheduler.step()

        image_to_text, text_to_image = get_test_results(model, data_loaders['val'])
        if image_to_text > best_acc:
            best_acc = image_to_text
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, model_path)
        
def get_test_results(model, data_loader):

    model.eval()
    image_to_text = []
    text_to_image = []
    
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = dim_reduce(model(inputs))
        distances = torch.cdist(targets, outputs)
        
        values, indices = torch.min(distances, 0)
        for n, i in enumerate(indices):
            text_to_image.append(1.0 if n == int(i) else 0.0)   
            
        values, indices = torch.min(distances, 1)
        for n, i in enumerate(indices):
            image_to_text.append(1.0 if n == int(i) else 0.0)     
        
    image_to_text = sum(image_to_text) / float(len(image_to_text))
    text_to_image = sum(text_to_image) / float(len(text_to_image))
    return image_to_text, text_to_image
    
if __name__ == "__main__":

    # Get arguments
    parser = argparse.ArgumentParser(description='CS2770 HW3')
    parser.add_argument('--epochs', type=int, default=25, help='The number of epochs')
    parser.add_argument('--model', type=str, default='alex', help='The type of CNN to use, either "alex" for AlexNet or "res" for Resnet18')
    parser.add_argument('--embedding', type=str, default='glove', help='The word embedding to use.  Must be "glove" or "w2v"')
    parser.add_argument('--image_data_set', type=str, default='coco', help='The image data set(s) to use.  Must be "coco" or "news"')
    parser.add_argument('--cross_domain_eval', action='store_true', help='Use this flag to do cross-domain evaluation on the data set not used for training')
    parser.add_argument('--data_dir', type=pathlib.Path, help='The directory where image and embedding pickle files can be found')
    parser.add_argument('--output_dir', type=pathlib.Path, help='Output')
    args = parser.parse_args()
    
    # Validate arguments
    if not (args.model == "alex" or args.model == "res"):
        raise Exception('Expected "alex" or "res" as model, found {args.model}')
    
    if not (args.embedding == "glove" or args.embedding == "w2v"):
        raise Exception('Expected "glove" or "w2v" as embedding, found {args.embedding}')
    
    if not (args.image_data_set == "coco" or args.image_data_set == "news"):
        raise Exception('Expected "coco" or "news" as image data set, found {args.image_data_set}')
              
    if not os.path.exists(args.data_dir):
        raise Exception('Not a valid path to find image and embedding pickle files: {args.data_dir}')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create parameters and model
    model_path = os.path.join(args.output_dir, f'{args.model}_{args.embedding}_{args.epochs}_{args.image_data_set}.pth')
    batch_size = 128
    num_workers = 2
    model = models.alexnet(pretrained=True) if args.model == "alex" else models.resnet18(pretrained=True)
    
    # Training
    print(f'Training with model {args.model}, embedding {args.embedding}, image data set {args.image_data_set}')
    data_loaders = get_loaders(args.data_dir, args.image_data_set, args.embedding, batch_size, num_workers)
    train(args.epochs, model, data_loaders, model_path)
    
    # Testing
    print(f'Testing with model {args.model}, embedding {args.embedding}, image data set {args.image_data_set}')
    model.load_state_dict(torch.load(model_path))
    image_to_text, text_to_image = get_test_results(model, data_loaders['test'])
    print(f'Model accuracy: image-to-text {image_to_text}; text-to-image {text_to_image}')
    
    # Cross-Domain Testing
    if args.cross_domain_eval:
        img_data_set = "news" if args.image_data_set == "coco" else "coco"
        print(f'Testing with model {args.model}, embedding {args.embedding}, image data set {img_data_set}')
        cd_data_loader = get_loaders(args.data_dir, img_data_set, args.embedding, batch_size, num_workers)['train']
        image_to_text, text_to_image = get_test_results(model, cd_data_loader)
        print(f'Model accuracy: image-to-text {image_to_text}; text-to-image {text_to_image}')   