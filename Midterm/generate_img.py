import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Any, Optional


class NoiseGenerator:
    """Class to generate different types of noise on images."""
    
    @staticmethod
    def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 25) -> np.ndarray:
        """
        Add Gaussian noise to an image.
        
        Args:
            image: Input image as numpy array
            mean: Mean of the Gaussian distribution
            sigma: Standard deviation of the Gaussian distribution
            
        Returns:
            Noisy image as uint8 numpy array
        """
        row, col, ch = image.shape
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_salt_pepper_noise(image: np.ndarray, salt_prob: float = 0.02, 
                              pepper_prob: float = 0.02) -> np.ndarray:
        """
        Add salt and pepper noise to an image.
        
        Args:
            image: Input image as numpy array
            salt_prob: Probability of salt noise (white pixels)
            pepper_prob: Probability of pepper noise (black pixels)
            
        Returns:
            Noisy image as uint8 numpy array
        """
        noisy = np.copy(image)
        
        # Salt noise (white pixels)
        salt_mask = np.random.random(image.shape) < salt_prob
        noisy[salt_mask] = 255
        
        # Pepper noise (black pixels)
        pepper_mask = np.random.random(image.shape) < pepper_prob
        noisy[pepper_mask] = 0
        
        return noisy
    
    @staticmethod
    def add_speckle_noise(image: np.ndarray, var: float = 0.05) -> np.ndarray:
        """
        Add speckle (multiplicative) noise to an image.
        
        Args:
            image: Input image as numpy array
            var: Variance of the Gaussian distribution
            
        Returns:
            Noisy image as uint8 numpy array
        """
        row, col, ch = image.shape
        gauss = np.random.normal(0, var**0.5, (row, col, ch))
        noisy = image + image * gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_poisson_noise(image: np.ndarray) -> np.ndarray:
        """
        Add Poisson noise to an image (simulates photon counting noise).
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Noisy image as uint8 numpy array
        """
        # Convert to float for calculation
        img_float = image.astype(float) / 255.0
        # Apply Poisson noise
        noisy_float = np.random.poisson(img_float * 255) / 255.0
        # Convert back to uint8
        return np.clip(noisy_float * 255, 0, 255).astype(np.uint8)


class NoiseDataset:
    """Class to create and manage noisy image datasets."""
    
    def __init__(self, image_paths: List[str], output_dir: str = "output_images"):
        """
        Initialize with paths to original images.
        
        Args:
            image_paths: List of paths to original images
            output_dir: Directory to save processed images
        """
        self.image_paths = image_paths
        self.noise_generator = NoiseGenerator()
        self.dataset = {}
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_image(self, path: str) -> np.ndarray:
        """
        Load an image from path and convert to RGB.
        
        Args:
            path: Path to the image file
            
        Returns:
            Image as RGB numpy array
            
        Raises:
            ValueError: If image cannot be loaded
        """
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not load image from {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def save_image(self, image: np.ndarray, img_name: str, noise_type: str) -> str:
        """
        Save an image to disk.
        
        Args:
            image: Image to save
            img_name: Base name of the image
            noise_type: Type of noise applied ('original', 'gaussian', etc.)
            
        Returns:
            Path where the image was saved
        """
        # Create image-specific directory
        img_dir = os.path.join(self.output_dir, img_name)
        os.makedirs(img_dir, exist_ok=True)
        
        # Save path
        save_path = os.path.join(img_dir, f"{noise_type}.png")
        
        # Convert RGB to BGR for OpenCV
        cv2_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, cv2_image)
        
        return save_path
    
    def create_dataset(self, save_images: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Create a dataset with original and noisy versions of each image.
        
        Args:
            save_images: Whether to save images to disk
            
        Returns:
            Dictionary with image names as keys and dictionaries of original/noisy images as values
        """
        for path in self.image_paths:
            img_name = os.path.basename(path).split('.')[0]
            try:
                original = self.load_image(path)
                
                # Generate noisy versions
                gaussian = self.noise_generator.add_gaussian_noise(original)
                salt_pepper = self.noise_generator.add_salt_pepper_noise(original)
                speckle = self.noise_generator.add_speckle_noise(original)
                poisson = self.noise_generator.add_poisson_noise(original)
                
                # Store in dataset
                self.dataset[img_name] = {
                    'original': original,
                    'gaussian': gaussian,
                    'salt_pepper': salt_pepper,
                    'speckle': speckle,
                    'poisson': poisson
                }
                
                # Save images if requested
                if save_images:
                    # Save original and each noise type separately
                    self.save_image(original, img_name, "original")
                    self.save_image(gaussian, img_name, "gaussian")
                    self.save_image(salt_pepper, img_name, "salt_pepper")
                    self.save_image(speckle, img_name, "speckle")
                    self.save_image(poisson, img_name, "poisson")
                    
                    print(f"Saved images for {img_name} in {os.path.join(self.output_dir, img_name)}")
                    
            except ValueError as e:
                print(f"Error processing {path}: {e}")
        
        return self.dataset

def main():
    """Main function to demonstrate usage."""
    # Example usage with actual image paths (modify as needed)
    image_paths = [
        'data/input/1/original.png'
    ]
    
    # Create dataset with separate image saving
    dataset_creator = NoiseDataset(image_paths, output_dir="data/input")
    noisy_dataset = dataset_creator.create_dataset(save_images=True)

if __name__ == "__main__":
    main()