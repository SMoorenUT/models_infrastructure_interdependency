import os


def check_folders_complete(directory):
    max_files = 0
    folder_files_count = {}

    # First pass to determine the maximum number of files in any folder
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            num_files = len(os.listdir(folder_path))
            folder_files_count[folder_name] = num_files
            if num_files > max_files:
                max_files = num_files

    # Second pass to check completeness based on the maximum number of files
    incomplete_folders = []

    for folder_name, num_files in folder_files_count.items():
        if num_files != max_files:
            incomplete_folders.append((folder_name, num_files))

    if incomplete_folders:
        for folder_name, num_files in incomplete_folders:
            print(f"Folder '{folder_name}' is incomplete (contains {num_files} files).")
    else:
        print("All folders are complete.")


if __name__ == "__main__":
    directory = "/media/p-drive/ET/CME/Current/Sander Mooren/scenarios_ema_1000/Output/"
    check_folders_complete(directory)
