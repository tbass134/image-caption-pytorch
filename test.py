import torch
import torchvision.transforms as transforms
import model
import utils
import dataset
image = "../../a-PyTorch-Tutorial-to-Image-Captioning/1500-1000-max.jpg"

transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

embed_size = 256
hidden_size = 256
vocab_size = len(dataset.vocab)
num_layers = 1
lr = 3e-4
num_epochs = 100

train_loader, dataset = get_loader("flickr8k/images/", "flickr8k/captions.txt", transform=transform)
model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to("cpu")
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=lr)

step = utils.load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
model.eval()

test_img1 = transform(Image.open(image).convert("RGB")).unsqueeze(0)
utils.print_examples(model, optimizer, "cpou")