import os

def change_extensions(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path):
            base_name, _ = os.path.splitext(file_name)
            new_file_name = f"{base_name}.png"
            new_file_path = os.path.join(folder_path, new_file_name)

            os.rename(file_path, new_file_path)
            print(f"Renamed: {file_name} -> {new_file_name}")


def count_jpg_files(directory):
    jpg_files = [file for file in os.listdir(directory) if file.lower().endswith('.png')]
    return len(jpg_files)

folder_path = "/data/yskim/LAION-Aesthetics-V2-6.5plus/data"  
change_extensions(folder_path)
print("All file extensions have been changed to .png.")

jpg_count = count_jpg_files(folder_path)
print(f"The folder contains {jpg_count} .png files.")