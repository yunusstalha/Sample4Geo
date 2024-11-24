import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class VigorPlus(Dataset):
    def __init__(self, dataset_root, cities_to_include, transform=None, val=False, is_satellite=True):
        """
        Args:
            dataset_root (str): Root directory of the dataset containing city folders.
            cities_to_include (list): List of city names to include in the dataset.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.dataset_root = dataset_root
        self.cities = cities_to_include
        self.transform = transform
        self.image_caption_pairs = []
        
        for city in self.cities:
            city_folder = os.path.join(self.dataset_root, city)
            if is_satellite:
                images_folder = os.path.join(city_folder, 'satellite')  # Adjust the folder name if different
                csv_file = os.path.join(city_folder, 'satellite_captions.csv')  # Adjust the CSV file name if different
            else:
                images_folder = os.path.join(city_folder, 'panorama')
                csv_file = os.path.join(city_folder, 'panorama_captions.csv')
            
            # Read the CSV file
            if val:
                df = pd.read_csv(csv_file).sample(frac=1).reset_index()
            else:
                df = pd.read_csv(csv_file)

            for idx, row in df.iterrows():
                image_filename = row['filename']
                caption = row['caption']
                image_path = os.path.join(images_folder, image_filename)
                
                # Check if the image file exists
                if os.path.isfile(image_path):
                    self.image_caption_pairs.append((image_path, caption))
                else:
                    print(f"Warning: Image file {image_path} not found.")
    
    def __len__(self):
        return len(self.image_caption_pairs)
    
    def __getitem__(self, idx):
        image_path, caption = self.image_caption_pairs[idx]
        # Open the image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, caption

class VigorCombinedDataset(Dataset):
    def __init__(self, dataset_root, cities_to_include, transform, val=False):
        """
        Args:
            dataset_root (str): Root directory of the dataset containing city folders.
            cities_to_include (list): List of city names to include in the dataset.s.
            val (bool, optional): Whether to use the validation split.
        """
        self.dataset_root = dataset_root
        self.cities = cities_to_include
        self.transform = transform


        self.image_caption_pairs = []

        for city in self.cities:
            city_folder = os.path.join(self.dataset_root, city)

            # Satellite images
            images_folder_sat = os.path.join(city_folder, 'satellite')
            csv_file_sat = os.path.join(city_folder, 'satellite_captions.csv')

            # Panorama images
            images_folder_pan = os.path.join(city_folder, 'panorama')
            csv_file_pan = os.path.join(city_folder, 'panorama_captions.csv')

            # Read satellite CSV file
            df_sat = pd.read_csv(csv_file_sat)
            if val:
                df_sat = df_sat.sample(frac=1).reset_index(drop=True)

            for idx, row in df_sat.iterrows():
                image_filename = row['filename']
                caption = row['caption']
                image_path = os.path.join(images_folder_sat, image_filename)

                # Check if the image file exists
                if os.path.isfile(image_path):
                    self.image_caption_pairs.append((image_path, caption, 'satellite'))
                else:
                    print(f"Warning: Satellite image file {image_path} not found.")

            # Read panorama CSV file
            df_pan = pd.read_csv(csv_file_pan)
            if val:
                df_pan = df_pan.sample(frac=1).reset_index(drop=True)

            for idx, row in df_pan.iterrows():
                image_filename = row['filename']
                caption = row['caption']
                image_path = os.path.join(images_folder_pan, image_filename)

                # Check if the image file exists
                if os.path.isfile(image_path):
                    self.image_caption_pairs.append((image_path, caption, 'panorama'))
                else:
                    print(f"Warning: Panorama image file {image_path} not found.")

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        image_path, caption, image_type = self.image_caption_pairs[idx]
        # Open the image
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, caption


if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt
    import textwrap
    import torch

    # Function to plot and save images with captions from a DataLoader
    def plot_images_with_captions_from_loader(data_loader, num_samples=5, save_path='sample_images_from_loader.png'):
        # Select a batch of images and captions
        for images, captions in data_loader:
            # Choose random indices from the batch
            num_samples = min(num_samples, len(images))
            indices = random.sample(range(len(images)), num_samples)

            # Create a figure to plot the images
            fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 4, 3))  # Adjust figure width per image
            plt.subplots_adjust(wspace=0.3)  # Adjust horizontal spacing between images

            for i, idx in enumerate(indices):
                image = images[idx]
                caption = captions[idx]

                # Convert image tensor to numpy array and transpose for plotting
                image = image.permute(1, 2, 0).numpy()

                # Ensure values are in range [0, 1] for displaying
                image = (image - image.min()) / (image.max() - image.min())

                # Wrap long captions for display
                wrapped_caption = "\n".join(textwrap.wrap(caption, width=60))  # Adjust width to limit text lines


                # Plot image and caption
                ax = axes[i]
                ax.imshow(image)
                ax.axis('off')
                ax.text(0.5, -0.5, wrapped_caption, ha='center', fontsize=8, transform=ax.transAxes)  # Move caption below
                ax.set_title("")  # Remove the title if you use this approach


            # Save the plot
            plt.tight_layout()
            plt.savefig(save_path)
            plt.show()

            # Break after the first batch since we're only interested in plotting a few samples
            break


    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
    ])

    dataset_root = '/home/erzurumlu.1/yunus/research_drive/data/VIGOR'
    cities_to_include = ['NewYork', 'Chicago']
    dataset = VigorPlus(dataset_root, cities_to_include, transform=transform, is_satellite=False)
    print(f"Number of samples in the dataset: {len(dataset)}")
    sample_image, sample_caption = dataset[0]
    print(f"Sample image size: {sample_image.size()}")
    print(f"Sample caption: {sample_caption}")
    from torch.utils.data import DataLoader

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    plot_images_with_captions_from_loader(data_loader)

    for images, captions in data_loader:
        print(f"Shape of the batch of images: {images.shape}")
        
        break