import argparse, os, pathlib, torch
from torchvision import transforms
from data_loader import get_loaders
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
	print(f'x1: {x1}')
	x2 = torch.stack([positive, negative], 0)
	print(f'x2: {x2}')
	distance = list(torch.cdist(x1, x2))
	print(f'distance: {distance}')
	pos_dist = float(distance[0])
	print(f'pos distance: {pos_dist}')
	neg_dist = float(distance[1])
	print(f'neg distance: {neg_dist}')
	return max(pos_dist - neg_dist + margin, 0)

# Get data loaders
data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

batch_size = 128
num_workers = 2
#data_loaders = get_loaders(args.data_dir, args.json_file, args.embedding_file, data_transforms, batch_size, True, num_workers)

a = torch.rand(50)
p = torch.rand(50)
n = torch.rand(50)

l = triplet_loss(a, p, n)


#model = EncoderCNN()
#model.to(device)


       
"""

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=config['learning_rate'])

# Train the models
total_step = len(data_loader)
for epoch in range(1,config['num_epochs']+1):
    for i, (images, questions, lengths) in enumerate(data_loader):
        
        # Set mini-batch dataset
        images = images.to(device)
        questions = questions.to(device)
        targets = pack_padded_sequence(questions, lengths, batch_first=True)[0]
        
        # Forward, backward and optimize
        features = encoder(images)
        outputs = decoder(features, questions, lengths)
        loss = criterion(outputs, targets)
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()

        # Print log info
        if i % config['log_step'] == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                  .format(epoch, config['num_epochs'], i, total_step, loss.item(), np.exp(loss.item()))) 
        
    torch.save(decoder.state_dict(), os.path.join(config['model_dir'], f'decoder-{epoch}.pth'))
    torch.save(encoder.state_dict(), os.path.join(config['model_dir'], f'encoder-{epoch}.pth'))

"""