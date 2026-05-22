import os
import json
from google.cloud import storage
from configs import config

class ArxivUnifiedEngine:
    def __init__(self):
        """Initializes the engine, connects to GCS, and sets up state tracking."""
        self.download_dir = config.DOWNLOAD_DIR
        self.state_file = config.STATE_FILE
        self.base_prefix = config.GCS_BASE_PREFIX
        os.makedirs(self.download_dir, exist_ok=True)
        
        # Connect to GCS anonymously (Zero billing)
        self.client = storage.Client.create_anonymous_client()
        self.bucket = self.client.bucket(config.GCS_BUCKET_NAME)
        
        # Discover all historical month folders and load the current state
        self.all_month_folders = self._get_all_month_prefixes()
        self.state = self._load_state()

    def _get_all_month_prefixes(self) -> list:
        """Scans the bucket and returns a sorted list of all YYMM folders."""
        print("Discovering historical arXiv folders...")
        iterator = self.client.list_blobs(self.bucket, prefix=self.base_prefix, delimiter='/')
        for _ in iterator: 
            pass # Exhaust iterator to populate prefixes
        
        # Extract the 'YYMM' part and sort chronologically
        prefixes = [p.split('/')[-2] for p in iterator.prefixes if p.endswith('/')]
        return sorted(prefixes)

    def _load_state(self) -> dict:
        """Loads the JSON cursor, or starts from the very first month if none exists."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
                
        first_month = self.all_month_folders[0] if self.all_month_folders else "2605"
        print(f"No state found. Starting historical backfill from month: {first_month}")
        return {"current_month": first_month, "last_file": ""}

    def _save_state(self) -> None:
        """Writes the cursor to disk to survive server crashes."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f)

    def get_batch(self, batch_size=50) -> list:
        """Pulls the next batch of PDFs. Rolls over to the next month automatically."""
        downloaded_paths = []
        current_month = self.state["current_month"]
        target_prefix = f"{self.base_prefix}{current_month}/"
        
        blobs = self.client.list_blobs(self.bucket, prefix=target_prefix)
        
        for blob in blobs:
            if not blob.name.endswith('.pdf'):
                continue
                
            filename = blob.name.split('/')[-1]
            
            # Skip files we've already processed
            if self.state["last_file"] and filename <= self.state["last_file"]:
                continue
                
            local_path = os.path.join(self.download_dir, filename)
            
            try:
                blob.download_to_filename(local_path)
                downloaded_paths.append(local_path)
                self.state["last_file"] = filename
                self._save_state()
            except Exception as e:
                print(f"Download failed for {filename}: {e}")
                continue
                
            if len(downloaded_paths) >= batch_size:
                return downloaded_paths
                
        # If the loop finishes without hitting batch_size, the month is complete.
        return self._rollover_to_next_month(downloaded_paths, batch_size)

    def _rollover_to_next_month(self, current_batch, batch_size) -> list:
        """Moves the cursor to the next chronological month folder."""
        try:
            current_index = self.all_month_folders.index(self.state["current_month"])
            next_month = self.all_month_folders[current_index + 1]
            
            print(f"\n--- Finished {self.state['current_month']}. Rolling over to {next_month} ---\n")
            self.state["current_month"] = next_month
            self.state["last_file"] = ""
            self._save_state()
            
            # Recursively fill the rest of the batch from the new month
            remaining_needed = batch_size - len(current_batch)
            if remaining_needed > 0:
                current_batch.extend(self.get_batch(batch_size=remaining_needed))
                
            return current_batch
            
        except IndexError:
            # Reached the edge of the dataset. Re-fetch folders to check for new months.
            self.all_month_folders = self._get_all_month_prefixes()
            
            if self.state["current_month"] in self.all_month_folders:
                new_index = self.all_month_folders.index(self.state["current_month"])
                if new_index < len(self.all_month_folders) - 1:
                    return self._rollover_to_next_month(current_batch, batch_size)
                
            # Fully caught up to today.
            return current_batch

    def cleanup_batch(self, file_paths: list) -> None:
        """Deletes the PDFs from local disk to prevent storage overflow."""
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print(f"Cleanup failed for {path}: {e}")