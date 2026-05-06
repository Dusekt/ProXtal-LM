import h5py
import numpy as np

def calculate_dataset_statistics(h5_file_path: str):
    """
    Parses the CrystalContactsDataset HDF5 file and prints
    comprehensive statistics for the thesis methodology section.
    """
    total_proteins = 0
    total_polymorphs = 0
    sequence_lengths = []
    polymorph_counts = []
    
    with h5py.File(h5_file_path, "r") as f:
        total_proteins = int(f.attrs.get("num_samples", len(f.keys())))
        
        for i in range(total_proteins):
            group_key = str(i)
            if group_key not in f:
                continue
                
            group = f[group_key]
            
            # 1. Get sequence length from the embedding tensor [L, D]
            if "embedding" in group:
                L = group["embedding"].shape[0]
                sequence_lengths.append(L)
                
            # 2. Count polymorphs (contact_0, contact_1, etc.)
            n_poly = sum(
                1 for k in group.keys() 
                if k.startswith("contact_") and not k.startswith("contact_og_")
            )
            # Dataset falls back to 1 if missing, but let's count actual data
            n_poly = max(n_poly, 1) 
            
            total_polymorphs += n_poly
            polymorph_counts.append(n_poly)

    # Calculate distributions
    seq_lengths = np.array(sequence_lengths)
    poly_counts = np.array(polymorph_counts)
    
    print("=" * 50)
    print("📊 DATASET STATISTICS FOR THESIS")
    print("=" * 50)
    print(f"Total Unique Proteins (Monomers): {total_proteins:,}")
    print(f"Total Crystal Polymorphs:         {total_polymorphs:,}")
    print(f"Average Polymorphs per Protein:   {np.mean(poly_counts):.2f}")
    print(f"Max Polymorphs for a single seq:  {np.max(poly_counts)}")
    print("-" * 50)
    print("Sequence Length Distribution:")
    print(f"  Minimum Length: {np.min(seq_lengths)} residues")
    print(f"  Maximum Length: {np.max(seq_lengths)} residues")
    print(f"  Mean Length:    {np.mean(seq_lengths):.1f} residues")
    print(f"  Median Length:  {np.median(seq_lengths):.1f} residues")
    print("=" * 50)
    
    # Polymorph distribution breakdown
    print("Polymorph Frequency Breakdown:")
    unique_counts, freq = np.unique(poly_counts, return_counts=True)
    for count, f in zip(unique_counts, freq):
        print(f"  Proteins with {count} polymorph(s): {f:,} ({f/total_proteins*100:.1f}%)")

# Example usage:
# calculate_dataset_statistics("path_to_your_train_dataset.h5")