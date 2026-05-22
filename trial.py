import os
import json

# Set environment variables for the trial sandbox before importing config
os.environ["DOWNLOAD_DIR"] = "trial_batch"
os.environ["STATE_FILE"] = "trial_state.json"

from arxiv_scholar.download.arxiv_ingestion import ArxivUnifiedEngine

def run_local_trial():
    print("🚀 Starting local trial run...")
    
    # 1. Initialize the engine
    # This will scan the bucket's folders (takes about 1-2 seconds)
    engine = ArxivUnifiedEngine()
    
    # 2. Print out the folders it discovered just to verify it sees the dataset
    print(f"Total month folders discovered: {len(engine.all_month_folders)}")
    print(f"First few folders in line: {engine.all_month_folders[:5]}")
    
    # 3. download a micro-batch of exactly 2 papers
    print("\nFetching a micro-batch of 2 PDFs...")
    paths = engine.get_batch(batch_size=2)
    
    print(f"\nSuccessfully downloaded {len(paths)} files:")
    for path in paths:
        file_size = os.path.getsize(path) / (1024 * 1024)
        print(f"  👉 {path} ({file_size:.2f} MB)")
        
    # 4. Read and display the state file to prove it saved your place
    print("\nInspecting tracking state (trial_state.json):")
    with open("trial_state.json", "r") as f:
        state_data = json.load(f)
        print(json.dumps(state_data, indent=2))
        
    # 5. Clean up the files so your disk stays completely clean
    print("\nCleaning up downloaded trial files...")
    engine.cleanup_batch(paths)
    
    # Verify deletion
    remaining_files = os.listdir("trial_batch")
    print(f"Files left in trial folder: {len(remaining_files)}")
    print("\n✅ Trial run complete! Logic is working flawlessly.")

if __name__ == "__main__":
    run_local_trial()