# models/archotecture/genomic_model_zero.py

'''
File containing model architecture
'''


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# from utils import find_common_adapters

# --------------------
# Dataloader
# --------------------

class DNADataset(Dataset):
    def __init__(self, filepath, is_test=True, header=0):
        self.data = pd.read_csv(filepath, sep='\t', header=header)
        self.is_test = is_test
        # Mapping constants 
        self.mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
    def one_hot_encode(self, seq):
        # conversion: string -> signs tables
        seq_array = np.frombuffer(seq.encode('ascii'), dtype=np.int8)

        # initialization zeros matrix
        encoded = np.zeros((4, len(seq_array)), dtype=np.float32)

        # Creating zero matrix (4 channels x sequence length)
        # we create masks and map 1.0 in right places 
        encoded[0, seq_array == ord('A')] = 1.0
        encoded[1, seq_array == ord('C')] = 1.0
        encoded[2, seq_array == ord('G')] = 1.0
        encoded[3, seq_array == ord('T')] = 1.0

        return encoded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # numpy -> torch tensor 
        sequence = self.one_hot_encode(row['sequence'])
        sequence = torch.from_numpy(sequence)
        
        # convert data type 

        # if dataset is treated as test it does not return targets 
        if self.is_test:
            seq_id = str(row['seq_id'])
            return seq_id, torch.tensor(sequence)
        
        # Targets
        ratio = torch.tensor(row['rna_dna_ratio'], dtype=torch.float32)
        active = torch.tensor(row['is_active'], dtype=torch.float32)
        
        return sequence.detach().clone(), ratio, active
    

class DNADatasetNoAdapters(DNADataset):
    def __init__(self, filepath, is_test=True, header=0):
        # Initialize the base DNADataset (loads self.data and self.mapping)
        super().__init__(filepath, is_test=is_test, header=header)
        
        # Find local adapters globally 
        self.prefix, self.suffix = find_common_adapters(self.data['sequence'])
        self.prefix_len = len(self.prefix)
        self.suffix_len = len(self.suffix)
        
        # Print info to verify everything is correct
        print(f"--- Dataset Initialized with Adapter Trimming ---")
        print(f"Detected Prefix: '{self.prefix}' ({self.prefix_len} bp)")
        print(f"Detected Suffix: '{self.suffix}' ({self.suffix_len} bp)")
        
    def __getitem__(self, idx):
        # Get the original output from the parent class
        # (Handling one-hot encoding and tensor conversion there)
        result = super().__getitem__(idx)
        
        if self.is_test:
            # result is (seq_id, sequence_tensor)
            seq_id, sequence = result
            # Trim the sequence: [channels, start:end]
            end_idx = sequence.shape[1] - self.suffix_len if self.suffix_len > 0 else sequence.shape[1]
            trimmed_seq = sequence[:, self.prefix_len : end_idx]
            return seq_id, trimmed_seq
        else:
            # result is (sequence_tensor, ratio, active)
            sequence, ratio, active = result
            # Trim the sequence: [channels, start:end]
            end_idx = sequence.shape[1] - self.suffix_len if self.suffix_len > 0 else sequence.shape[1]
            trimmed_seq = sequence[:, self.prefix_len : end_idx]
            return trimmed_seq, ratio, active

# --------------------
# Model Zero - Basic architecture only to test model
# --------------------

class GenomicModelZero(nn.Module):
    def __init__(self):
        super(GenomicModelZero, self).__init__()
        
        # Convolution layers
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=64, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10, padding=5),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(8) # Sprowadza wyjście do stałego rozmiaru
        )
        
        # Dense layer (Common)
        self.flatten = nn.Flatten()
        self.fc_shared = nn.Sequential(
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Heads 
        ## classification
        self.classification_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        ## regression
        self.regression_head = nn.Sequential(
            nn.Linear(256, 1) # Linear - ratio can be negative 
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        shared_features = self.fc_shared(x)
        
        is_active = self.classification_head(shared_features)
        ratio = self.regression_head(shared_features)
        
        return is_active.view(-1), ratio.view(-1)
    

# --------------------
# Model One  Here I introduce different forward method adjusting complementary sequence 
# --------------------

class GenomicModelOne(nn.Module):
    '''
    In this model we force model to learn complementary sequences
    '''

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model # zero model loaded

    def forward(self, x):
        # original sequence
        out_active_fwd, out_ratio_fwd = self.base_model(x)

        # Reverse complement 
        rc_x = x.flip(dims=[-1, -2])
        out_active_rev, out_ratio_rev = self.base_model(rc_x)

        # We aggregate results 
        final_active = (out_active_fwd + out_active_rev) / 2
        final_ratio = (out_ratio_fwd + out_ratio_rev) / 2

        return final_active, final_ratio

    

# --------------------
# DeepSTARR replication - much to complicated for problem 
# --------------------

class GenomicModelDeepStarr(nn.Module):
    '''
    In this model we use architecture from DeepSTARR including 
    '''

    def __init__(self, sequence_length=200):
        super(GenomicModelDeepStarr, self).__init__()

        # it is strictly reproduce part from DeepSTARR
        # Filters amount: 246, 60, 60, 120 | Kernel sizes: 7, 3, 5, 3
        filt_sizes = [246, 60, 60, 120]
        kernel_sizes = [7, 3, 5, 3]

        layers = []
        in_channels = 4 # 4 bases

        # 
        for out_channels, k_size in zip(filt_sizes, kernel_sizes):
            layers.append(nn.Conv1d(
                in_channels=in_channels, 
                out_channels=out_channels,
                kernel_size=k_size,
                padding='same'
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        self.flatten_size = 120 * (sequence_length // (2**4))

        self.fc_shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(256, 64),
            # applied nonlinearity 
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    
    
    def forward(self, x):
        '''
        Here we implement reverse complement
        '''
        out_active_fwd, out_ratio_fwd = self.basic_forward(x)

        # Reverse complement 
        rc_x = x.flip(dims=[-1, -2])
        out_active_rev, out_ratio_rev = self.basic_forward(rc_x)

        # We aggregate results 
        final_active = (out_active_fwd + out_active_rev) / 2
        final_ratio = (out_ratio_fwd + out_ratio_rev) / 2

        return final_active, final_ratio

    def basic_forward(self, x):   
        x = self.conv_layers(x)
        x = self.fc_shared(x)
        
        is_active = self.classification_head(x)
        rna_dna_ratio = self.regression_head(x)
        
        return is_active.view(-1), rna_dna_ratio.view(-1)

# --------------------
# Model Two - this version had multiple modifications - it is to complex for problem so it does not generlize 
# --------------------

class GenomicModelTwo(nn.Module):
    '''
    We saw that DeepSTARR was to complex. I went back to ogirinal idea, ow increasing channels amount - my model could not learn any more pattern (besides null one).
    '''
    def __init__(self, sequence_length=200, last_max_pool_size=2):
        super(GenomicModelTwo, self).__init__()
        
        # Convolution layers
        self.conv_block = nn.Sequential(
            # first layers is looking for motifs
            nn.Conv1d(in_channels=4, out_channels=100, kernel_size=15, padding=7),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # sequence_length -> sequence_length/2
            # second layer for relation between them 
            nn.Conv1d(in_channels=100, out_channels=125, kernel_size=21, padding=10),
            nn.BatchNorm1d(125),
            nn.ReLU(),
            nn.MaxPool1d(2) # sequence_length/2 -> sequence_length/4
        )
        
        # Dense layer (Common)
        self.flatten = nn.Flatten()
        self.fc_shared = nn.Sequential(
            nn.Linear(125 * ((sequence_length // 2) // 2), 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Heads 
        ## classification
        self.classification_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        ## regression
        self.regression_head = nn.Sequential(
            nn.Linear(256, 64), # Linear - ratio can be negative 
            nn.ELU(),
            nn.Linear(64, 1)
        )


    # basic forward without reverse complement part
    def forward(self, x):   
        x = self.conv_block(x)
        x = x.view(x.size(0), -1) 
        x = self.fc_shared(x)
        
        is_active = self.classification_head(x)
        rna_dna_ratio = self.regression_head(x)
        
        return is_active.view(-1), rna_dna_ratio.view(-1)
    

class GenomicModelTwo_complement(nn.Module):
    '''
    In this model we force model to learn complementary sequences
    '''

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model # zero model loaded

    def forward(self, x):
        # original sequence
        out_active_fwd, out_ratio_fwd = self.base_model(x)

        # Reverse complement 
        rc_x = x.flip(dims=[-1, -2])
        out_active_rev, out_ratio_rev = self.base_model(rc_x)

        # We aggregate results 
        final_active = (out_active_fwd + out_active_rev) / 2
        final_ratio = (out_ratio_fwd + out_ratio_rev) / 2

        return final_active, final_ratio

# --------------------
# Zero model rebuild - here is most sense version for which we had not enough time to learn 
# --------------------

class GenomicModelZeroAdjusted(nn.Module):
    def __init__(self):
        super(GenomicModelZeroAdjusted, self).__init__()
        
        # Convolution layers
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=64, kernel_size=15, padding=7),
            # TODO - dodaje batche żęby nie wyrzucały współczynników za bardzo
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8) # TODO zmieniamy na średnią (na regresji może nie wychwytywać sygnału)
        )
        
        # Dense layer (Common)
        self.fc_shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Heads 
        ## classification
        self.classification_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        ## regression
        # TODO - zmieniamy regresje
        self.regression_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ELU(),            # ELU jest lepsze niż ReLU dla regresji (zakres ujemny)
            nn.Dropout(0.2),     # Dodatkowa ochrona przed overfittingiem w regresji
            nn.Linear(128, 1)
        )

    def forward(self, x):
        if self.training:
            # we change strategy to randomly flip sequence
            if torch.rand(1) > 0.5:
                x = x.flip(dims=[-1, -2])
            return self.basic_forward(x)
        else:
            # during validation we take mean values,
            out_active_fwd, out_ratio_fwd = self.basic_forward(x)
            rc_x = x.flip(dims=[-1, -2])
            out_active_rev, out_ratio_rev = self.basic_forward(rc_x)
            return (out_active_fwd + out_active_rev) / 2, (out_ratio_fwd + out_ratio_rev) / 2

    def basic_forward(self, x):   
        x = self.conv_block(x)
        x = self.fc_shared(x)
        
        is_active = self.classification_head(x)
        rna_dna_ratio = self.regression_head(x)
        
        return is_active.view(-1), rna_dna_ratio.view(-1)