# %%
import os
import re

# %%
def verify_virtis_attached_labels(archive_dir):
    print(f"Scanning {archive_dir} for attached-label .CAL and .GEO files...")
    
    # We use byte-strings (b"...") here because we are reading the file in binary mode.
    # This prevents UnicodeDecodeErrors when the read() buffer accidentally catches 
    # the beginning of the binary data payload right after the header.
    record_bytes_re = re.compile(rb"RECORD_BYTES\s*=\s*(\d+)")
    file_records_re = re.compile(rb"FILE_RECORDS\s*=\s*(\d+)")
    
    files_seen = 0
    files_checked = 0
    corrupt_files = []

    # followlinks=True ensures we don't get tripped up if the directory is a symlink mount
    for root, _, files in os.walk(archive_dir, followlinks=True):
        for file in files:
            ext = file.upper().split('.')[-1]
            
            if ext in ['CAL', 'GEO']:
                files_seen += 1
                file_path = os.path.join(root, file)
                
                # Print the first few matches so we know the traversal is working
                if files_seen <= 3:
                    print(f"[DEBUG] Found target file: {file_path}")

                try:
                    # Read the first 16KB. The PDS3 ASCII header is always at the top.
                    with open(file_path, 'rb') as f:
                        header = f.read(16384)
                        
                    rb_match = record_bytes_re.search(header)
                    fr_match = file_records_re.search(header)
                    
                    if rb_match and fr_match:
                        # In PDS3, the total expected file size is exactly:
                        # RECORD_BYTES * FILE_RECORDS (this includes the header itself)
                        expected_size = int(rb_match.group(1)) * int(fr_match.group(1))
                        actual_size = os.path.getsize(file_path)
                        
                        files_checked += 1
                        
                        if actual_size != expected_size:
                            print(f"\n[!] TRUNCATION DETECTED: {file_path}")
                            print(f"    Expected: {expected_size} bytes, Actual: {actual_size} bytes")
                            corrupt_files.append(file_path)
                            
                        # Keep us updated on progress
                        if files_checked % 500 == 0:
                            print(f"Verified {files_checked} files...")
                            
                    else:
                        print(f"\n[?] Warning: Missing RECORD_BYTES or FILE_RECORDS in header of {file_path}")
                        
                except PermissionError:
                    print(f"\n[!] Permission denied reading: {file_path}")
                except Exception as e:
                    print(f"\n[!] Error processing {file_path}: {e}")

    print("-" * 60)
    print(f"Total .CAL/.GEO files found: {files_seen}")
    print(f"Total files successfully verified: {files_checked}")
    
    if files_seen == 0:
        print("[!] ERROR: No .CAL or .GEO files were found. Double check the mount point or directory path.")
    elif files_checked == 0:
        print("[!] ERROR: Found files, but could not parse the PDS3 size keywords in the first 16KB.")
    elif corrupt_files:
        print(f"[!] Found {len(corrupt_files)} partial or corrupt downloads.")
        # Optional: dump bad files to a list for easy re-downloading
        with open("corrupt_files.list", "w") as f:
            for bad_file in corrupt_files:
                f.write(f"{bad_file}\n")
    else:
        print("Success! All checked files match their expected PDS3 byte counts. Data is structurally sound.")

# %%
if __name__ == "__main__":
    target_directory = "/exomars/data/external/venus/venus_express/VIRTIS/VIRTIS_M_calibrated_archive" 
    verify_virtis_attached_labels(target_directory)
# %%
