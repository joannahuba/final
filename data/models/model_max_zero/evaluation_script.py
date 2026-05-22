import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import classes from model.py as requested
from model import DNADataset, GenomicModelZero

def main():
    # Check if correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python evaluation_script.py <path_to_model> <path_to_test_data>")
        sys.exit(1)

    model_path = sys.argv[1]
    test_data_path = sys.argv[2]

    # Device selection logic (CUDA, MPS, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: GPU ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    # 1. Initialize architecture and load the trained model
    model = GenomicModelZero().to(device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Load state dict (check if it is a dictionary containing 'model_state_dict' or the state_dict itself)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Set model to evaluation mode
    model.eval()

    # 2. Read the test dataset (TSV) using the DNADataset class
    # is_test=True ensures it looks for 'id'/'seq_id' and 'sequence' without requiring targets
    try:
        test_dataset = DNADataset(test_data_path, is_test=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # 3. Generate predictions and 4. Print results to stdout in tab-separated format
    # Header required by task description
    print("id\tpredicted_is_active\tpredicted_rna_dna_ratio")

    with torch.no_grad():
        for batch in test_loader:
            # Handle the return format of DNADataset.__getitem__ for test mode
            # Based on user code: returns (seq_id, sequence_tensor)
            seq_ids, sequences = batch
            sequences = sequences.to(device)

            # Forward pass
            pred_active_probs, pred_ratios = model(sequences)

            # Post-processing:
            # - Classification: apply threshold 0.5 to Sigmoid output
            # - Regression: use raw continuous values
            predicted_is_active = (pred_active_probs > 0.5).int().cpu().numpy()
            predicted_rna_dna_ratio = pred_ratios.cpu().numpy()

            # Iterate through batch and print tab-separated values
            for i in range(len(seq_ids)):
                print(f"{seq_ids[i]}\t{predicted_is_active[i]}\t{predicted_rna_dna_ratio[i]}")

if __name__ == "__main__":
    main()
