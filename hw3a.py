import argparse, os, pathlib, random, torch
import torch.optim as optim
import torchvision.models as models
from torch.nn import AvgPool1d
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
    
# Calculate triplet loss for the given anchor, positive, and negative tensors
def triplet_loss(anchor, positive, negative, margin=0.5):
    x1 = anchor.unsqueeze(0)
    x2 = torch.stack([positive, negative], 0)
    distance = torch.cdist(x1, x2).tolist()[0]
    pos_dist = float(distance[0])
    neg_dist = float(distance[1])
    return max(pos_dist - neg_dist + margin, 0)

# For each output and each target, calculate the triplet loss from the target and a negative
# sample.  Return the average loss
def triplet_loss_batch(outputs, targets):

	output_list = list(outputs)
	target_list = list(targets)
	l = len(output_list)
	losses = []
	
	for i, (output, target) in enumerate(zip(output_list, target_list)):
		n = random.randrange(l)
		while n == i:
			n = random.randrange(l)
		
		losses.append(triplet_loss(output, target, target_list[n]))
	
	return torch.Tensor(losses)
		
def train(epochs, data_loaders):

    model = models.alexnet(pretrained=True)
    model.to(device)
    #summary(model, (3, 299, 299))

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

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
    
            loss = triplet_loss_batch(outputs, targets)
            loss.backward()
            optimizer.step()
    
        scheduler.step()
    
if __name__ == "__main__":

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
    
    # Create data loaders
    #batch_size = 128
    batch_size = 5
    num_workers = 2
    data_loaders = get_loaders(args.data_dir, args.json_file, args.embedding_file, batch_size, num_workers)
    
    # Train
    train(args.epochs, data_loaders)
    
    