from get_loader import get_loader
import torchvision.transforms as transforms
import pickle
transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(),]
)

train_loader, dataset = get_loader(
    root_folder="flickr8k/images",
    annotation_file="flickr8k/captions.txt",
    transform=transform,
    num_workers=2,
)

with open('vocab_itos.pkl', 'wb') as f:
    pickle.dump(dataset.vocab.itos, f)