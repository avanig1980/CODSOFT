import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
import numpy as np
import random

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Image Captioning Model with Encoder-Decoder Architecture
        
        Args:
            embed_size (int): Embedding dimension
            hidden_size (int): RNN hidden layer size
            vocab_size (int): Size of vocabulary
            num_layers (int): Number of RNN layers
        """
        super(ImageCaptioningModel, self).__init__()
        
        # Image Encoder (CNN Feature Extractor)
        self.encoder = models.resnet50(pretrained=True)
        # Remove the last fully connected layer
        modules = list(self.encoder.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        
        # Freeze CNN parameters
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        
        # Image Feature Projection
        self.image_feature_proj = nn.Linear(
            self.encoder[-1][-1].bn3.num_features, 
            embed_size
        )
        
        # Word Embedding Layer
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        
        # Decoder (RNN/LSTM)
        self.decoder = nn.LSTM(
            input_size=embed_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Output Layer (Vocabulary Prediction)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, images, captions, lengths):
        """
        Forward pass for training
        
        Args:
            images (tensor): Input images
            captions (tensor): Target captions
            lengths (tensor): Caption lengths
        
        Returns:
            tensor: Predicted outputs
        """
        # Extract image features
        image_features = self.encoder(images)
        image_features = image_features.reshape(image_features.size(0), -1)
        image_features = self.image_feature_proj(image_features)
        
        # Embed captions
        embeddings = self.word_embedding(captions)
        
        # Combine image features with caption embeddings
        features = torch.cat((image_features.unsqueeze(1), embeddings), dim=1)
        
        # Pack padded sequence
        packed = pack_padded_sequence(
            features, 
            lengths, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # Decode
        hiddens, _ = self.decoder(packed)
        
        # Generate output
        outputs = self.output_layer(hiddens[0])
        
        return outputs
    
    def generate_caption(self, image, vocab, max_length=20):
        """
        Generate caption for a single image
        
        Args:
            image (tensor): Input image
            vocab (dict): Vocabulary mapping
            max_length (int): Maximum caption length
        
        Returns:
            str: Generated caption
        """
        # Reverse vocabulary for decoding
        idx_to_word = {v: k for k, v in vocab.items()}
        
        # Extract image features
        with torch.no_grad():
            image_features = self.encoder(image.unsqueeze(0))
            image_features = image_features.reshape(image_features.size(0), -1)
            image_features = self.image_feature_proj(image_features)
        
        # Initialize generation
        sampled_ids = []
        inputs = image_features.unsqueeze(1)
        
        # Hidden state initialization
        hidden = None
        
        for _ in range(max_length):
            # Decode
            hiddens, hidden = self.decoder(inputs, hidden)
            outputs = self.output_layer(hiddens.squeeze(1))
            
            # Sample word
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted.item())
            
            # Break if end of sequence
            if predicted.item() == vocab['<END>']:
                break
            
            # Prepare next input
            inputs = self.word_embedding(predicted).unsqueeze(1)
        
        # Convert to words
        sampled_caption = [
            idx_to_word.get(idx, '<UNK>') 
            for idx in sampled_ids
        ]
        
        return ' '.join(sampled_caption)

class Vocabulary:
    """
    Vocabulary management for image captioning
    """
    def __init__(self, freq_threshold=5):
        self.itos = {0: '<PAD>', 1: '<START>', 2: '<END>', 3: '<UNK>'}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
    
    def build_vocabulary(self, captions):
        """
        Build vocabulary from training captions
        
        Args:
            captions (list): List of captions
        """
        frequencies = {}
        
        # Count word frequencies
        for caption in captions:
            for word in caption.lower().split():
                frequencies[word] = frequencies.get(word, 0) + 1
        
        # Add words meeting frequency threshold
        next_index = len(self.itos)
        for word, freq in frequencies.items():
            if freq > self.freq_threshold:
                self.stoi[word] = next_index
                self.itos[next_index] = word
                next_index += 1
    
    def numericalize(self, text):
        """
        Convert text to numeric tokens
        
        Args:
            text (str): Input text
        
        Returns:
            list: Numeric tokens
        """
        tokens = [self.stoi.get(word.lower(), self.stoi['<UNK>']) 
                  for word in text.split()]
        tokens = [self.stoi['<START>']] + tokens + [self.stoi['<END>']]
        return tokens

class ImageCaptioningDataLoader:
    """
    Custom data loader for image captioning
    """
    def __init__(self, image_paths, captions, vocab, transform=None):
        self.image_paths = image_paths
        self.captions = captions
        self.vocab = vocab
        
        # Default image transformations
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Prepare caption
        caption = self.captions[idx]
        numeric_caption = self.vocab.numericalize(caption)
        
        return image, torch.tensor(numeric_caption)

def train_image_captioning_model(
    image_paths, 
    captions, 
    embed_size=256, 
    hidden_size=512, 
    epochs=10, 
    learning_rate=1e-3
):
    """
    Train image captioning model
    
    Args:
        image_paths (list): Paths to training images
        captions (list): Corresponding captions
        embed_size (int): Embedding dimension
        hidden_size (int): RNN hidden size
        epochs (int): Training epochs
        learning_rate (float): Optimizer learning rate
    
    Returns:
        ImageCaptioningModel: Trained model
    """
    # Build vocabulary
    vocab = Vocabulary()
    vocab.build_vocabulary(captions)
    
    # Prepare data loader
    dataset = ImageCaptioningDataLoader(
        image_paths, 
        captions, 
        vocab
    )
    
    # Initialize model
    model = ImageCaptioningModel(
        embed_size=embed_size, 
        hidden_size=hidden_size, 
        vocab_size=len(vocab.stoi)
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate
    )
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        
        for image, caption in dataset:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                image.unsqueeze(0), 
                caption[:-1].unsqueeze(0), 
                [len(caption)-1]
            )
            
            # Compute loss
            loss = criterion(outputs, caption[1:])
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")
    
    return model

# Example usage
def main():
    # Sample image paths and captions (replace with your dataset)
    image_paths = [
        'path/to/image1.jpg', 
        'path/to/image2.jpg'
    ]
    captions = [
        'A dog playing in the park',
        'A cat sitting on a window sill'
    ]
    
    # Train model
    model = train_image_captioning_model(
        image_paths, 
        captions
    )
    
    # Example inference
    def load_image(image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        image = Image.open(image_path).convert('RGB')
        return transform(image)
    
    # Vocabulary instance from training
    vocab = Vocabulary()
    vocab.build_vocabulary(captions)
    
    # Generate caption for an image
    test_image_path = 'path/to/test_image.jpg'
    test_image = load_image(test_image_path)
    
    # Generate caption
    generated_caption = model.generate_caption(
        test_image, 
        vocab.stoi
    )
    print("Generated Caption:", generated_caption)

if __name__ == "__main__":
    main()