import argparse, os, pathlib, torch
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from data_loader import get_loaders
from torchsummary import summary

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='CS2770 HW3')
parser.add_argument('--epochs', type=int, default=25, help='The number of epochs')
parser.add_argument('--data_dir', type=pathlib.Path, help='The data set to use for training, testing, and validation')
parser.add_argument('--json_file', type=pathlib.Path, help='The json file with data set captions')
parser.add_argument('--embedding_file', type=pathlib.Path, help='The embedding file')
parser.add_argument('--output_dir', type=pathlib.Path, help='Output')
args = parser.parse_args()

# Create directories
if not os.path.exists(args.data_dir):
    os.makedirs(args.data_dir)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

def triplet_loss(anchor, positive, negative, margin=0.5):
    x1 = anchor.unsqueeze(0)
    x2 = torch.stack([positive, negative], 0)
    distance = torch.cdist(x1, x2).tolist()[0]
    pos_dist = float(distance[0])
    neg_dist = float(distance[1])
    return max(pos_dist - neg_dist + margin, 0)

# Get data loaders
data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

#batch_size = 128
batch_size = 3
num_workers = 2
data_loaders = get_loaders(args.data_dir, args.json_file, args.embedding_file, data_transforms, batch_size, True, num_workers)


model = models.alexnet(pretrained=True)
model.to(device)
#summary(model, (3, 299, 299))

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the models
for epoch in range(1,args.epochs+1):
    print(f'Epoch {epoch} of {args.epochs}')
    
    model.train()
    
    for inputs, targets in data_loaders['train']:
        
        # Set mini-batch dataset
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        #outputs = model(inputs)
        #print(f'Outputs: {outputs}')
        break
        
        #loss = triplet_loss(outputs, targets)
        #loss.backward()
        #optimizer.step()