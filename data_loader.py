#GOAL
    #Convert text to numerical data for model training
        #1. We need a vocabulary to map each word to an index
        #2. We need to setup a pytorch dataset to load the data
        #3. Setup padding of every batch (all examples should be of same seq_len and setup dataloader)
            #4. Padding fixes the issue that each scentence in the is a different size of words, 
        #       but pytorch needs them all to be the same length, so we add some sort of padding to make them equal
        #
#Imports
    #pandas: to find corresponding image + caption
    #spacy: tokenizer, can manipulate strings
    #pad_sequence: For padding mentioned above
    #Image: Loads images

import os 
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import torchvision.transforms as transforms

spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self,freq_threshold): #freq_threshold tells the pipline, if a word is not repeated this many amount of times, then dont include in the vocabulary
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"} #dictionary that maps an index with padding,start of captions, end of captions and unkowns
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>": 2, "<UNK>":3}
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return len(self.itos)
    
    #Takes a text and maps it to a list
    #"My dog is Amazing" -> ["my","dog","is","amazing"]
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self,sentence_list):
        frequencies = {}
        index = 4 #already have 4 indicies in itos and stoi
        
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] +=1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = index
                    self.itos[index] = word
                    index += 1
                    
    def numericalize(self,text):
        tokenized_text = self.tokenizer_eng(text)
        return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]
        


class FlickrDataset(Dataset):   
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold = 5):
            self.root_dir = root_dir #flicker8k/images
            self.df = pd.read_csv(captions_file) #reads caption file and turns it into pandas dataframe, then stored into df, like a table with rows and columns
            self.transform = transform
            
            #Make sure the image and caption columns are named correctly
            self.df["image"] = self.df["image"].astype(str).str.strip().apply(os.path.basename) #removes any leading or trailing spaces
            
            def _exists(fn):
                return os.path.exists(os.path.join(self.root_dir, fn))
            
            exists_mask = self.df["image"].apply(_exists)
            missing= int((~exists_mask).sum())
            if missing:
                print(f"[FlickrDataset] Warning: {missing} images are missing from {self.root_dir}. They will be ignored.")
                self.df = self.df[exists_mask].reset_index(drop=True)
            
            #Get image and columns
            self.imgs = self.df["image"]
            self.captions = self.df["caption"]
            
            #init vocabulary and build vocab
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocabulary(self.captions.tolist())
    
    #returns length of dataframe        
    def __len__(self):
        return len(self.df)
    
    #Getting a single item, caption and image together
    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = str(self.imgs[index]).strip()
        path = os.path.join(self.root_dir, img_id)

        # at this point it should exist; if not, raise loudly
        try:
            img = Image.open(path).convert("RGB")
        except FileNotFoundError:
            raise RuntimeError(f"Unexpected missing file after filtering: {img_id}")

        if self.transform is not None:
            img = self.transform(img)

        tokens = [self.vocab.stoi["<SOS>"]]
        tokens += self.vocab.numericalize(caption)
        tokens.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(tokens)
    
class MyCollate:
    def __init__(self,pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch] #Gives each 
        imgs = torch.cat(imgs, dim= 0) ## Combines all imgs into one batch
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets,batch_first=False,padding_value=self.pad_idx)
        
        return imgs,targets
    
def get_loader(
    root_folder,
    annotation_file,
    transform,
    batchsize =32,
    num_workers = 0,
    shuffle=True,
    pin_memory = True,
):        
       dataset = FlickrDataset(root_folder,annotation_file,transform=transform)
       
       pad_idx=dataset.vocab.stoi["<PAD>"]
       
       loader = DataLoader(
           dataset=dataset,
           batch_size=batchsize,
           num_workers=num_workers,
           shuffle=shuffle,
           pin_memory=pin_memory,
           collate_fn=MyCollate(pad_idx=pad_idx),
       )
       return loader
def main():
    transform = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]
    )  
    dataloader = get_loader("Flickr8k/Images/", annotation_file="Flickr8k/captions.txt", transform=transform)
    for idx, (imgs,captions) in enumerate(dataloader):
        print(imgs.shape)
        print(captions.shape)  

if __name__ == "__main__":
    main()   