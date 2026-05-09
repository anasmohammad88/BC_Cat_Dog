import torch
from torch.utils.data import Dataset, DataLoader 
from PIL import Image
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt

from cnn.model import CatDogClassification

CLASS_TO_IDX = {"cats": 0, "dogs": 1}
IDX_TO_CLASS = {value: key for key, value in CLASS_TO_IDX.items()}

class ImageDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = Path(dataset_path)
        self.samples = self.read_samples()
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        image_path, label =  self.samples[index]
        img_rgb = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(img_rgb)
        else:
            image = img_rgb
        return image, label
    
    def read_samples(self):
        samples = []
        for cls, label in CLASS_TO_IDX.items():
            class_dir = self.dataset_path/cls
            if not class_dir.exists(): raise ValueError(f"Dataset path incorrect: {class_dir}")
            
            for images_path in class_dir.glob('*'):
                if images_path.suffix.lower() == ".jpg": samples.append((str(images_path), label))
    
        if not samples: raise RuntimeError(f"No images found in {self.dataset_path}")
        return samples
    
def load_data(dataset_path, transform, batch_size=128, num_workers=2):
    image_dataset = ImageDataset(dataset_path = dataset_path, transform=transform)
    return DataLoader(dataset=image_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def show_sample(dataset):
    seen = set()
    for idx, (_, label) in enumerate(dataset.samples):
        if label in seen:
            continue

        image, _ = dataset[idx]
        if isinstance(image, torch.Tensor):
            img = image.permute(1, 2, 0).cpu().numpy()
        else:
            img = image

        plt.imshow(img)
        plt.title(f"class={label} | index={idx}")
        plt.axis("off")
        plt.show()

        seen.add(label)
        if len(seen) == len(CLASS_TO_IDX):
            break



def plot_prediction(model,img_path,transform,device="cpu"):
    model.eval()
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        prob_dog = torch.sigmoid(output).item()
        prob_cat = 1 - prob_dog

        if prob_dog >= 0.5:
            predicted_class = "dogs"
            confidence = prob_dog
        else:
            predicted_class = "cats"
            confidence = prob_cat

    plt.figure(figsize=(6, 6))

    plt.imshow(image)

    plt.title(f"Prediction: {predicted_class}\n"f"Confidence: {confidence * 100:.2f}%")

    plt.axis("off")
    plt.show()


def save_model(model, name="model"):
    path = Path("checkpoints") / f"{name}.pt"
    path.parent.mkdir(exist_ok=True)

    torch.save(model.state_dict(), path)
    return path


def load_model(model, name="model", device="cpu"):
    path = Path("checkpoints") / f"{name}.pt"

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    return model
    
    
if __name__ == "__main__":
    # dataset_path="data/DogsVsCats/train"
    transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
    # dataset = ImageDataset(dataset_path=dataset_path,transform=transform)
    # show_sample(dataset)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CatDogClassification()
    model = load_model(model, name="BC_cat_dog", device=device)
    model.to(device)
    
    # plot_prediction(model=model,img_path="data/DogsVsCats/test/dogs/dog.191.jpg",transform=transform,device=device)
    plot_prediction(model=model,img_path="data/DogsVsCats/test/cats/cat.108.jpg",transform=transform,device=device)