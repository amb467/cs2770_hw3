import argparse, os, pathlib, torch
from torchvision import transforms
from data_loader import get_loaders

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='CS2770 HW2')
parser.add_argument('--epochs', type=int, default=25, help='The number of epochs')
parser.add_argument('--data_dir', type=pathlib.Path, help='The data set to use for training, testing, and validation')
parser.add_argument('--json_file', type=pathlib.Path, help='The json file with data set captions')
parser.add_argument('--embedding_file', type=pathlib.Path, help='The embedding file')
args = parser.parse_args()

data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

batch_size = 128
num_workers = 2
data_loaders = get_loaders(args.data_dir, args.json_file, args.embedding_file, data_transforms, batch_size, True, num_workers)
    
"""

encoder, decoder, data_loader, config = proc(args, 'train', root_dir, 'train.py')

# Create model directory
if not os.path.exists(config['model_dir']):
    os.makedirs(config['model_dir'])

# Put models on device
encoder = encoder.to(device)
decoder = decoder.to(device)

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