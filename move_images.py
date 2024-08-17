import os
import shutil
import logging

# Configure logging
logging.basicConfig(
    filename='move_images.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Source and target directories
source_dir = 'classifier'  # Corrected directory name
target_dir_train = 'training_set'
target_dir_test = 'testing_set'

# Check if source directory exists
if not os.path.exists(source_dir):
    logging.error("Source directory '%s' does not exist.", source_dir)
    exit()

logging.info("Source directory: %s", source_dir)

# Create target directories if they don't exist
os.makedirs(target_dir_train, exist_ok=True)
os.makedirs(target_dir_test, exist_ok=True)

logging.info("Target directories '%s' and '%s' created or already exist.", target_dir_train, target_dir_test)

# Process files in the source directory
files_moved = 0
for filename in os.listdir(source_dir):
    if filename.endswith('.jpg'):
        logging.info("Processing file: %s", filename)

        src_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(target_dir_train, filename)  # Example logic

        logging.debug("Source path: %s", src_path)
        logging.debug("Destination path: %s", dest_path)

        try:
            shutil.move(src_path, dest_path)
            logging.info("Moved '%s' to '%s'", filename, target_dir_train)
            files_moved += 1
        except Exception as e:
            logging.error("Error moving file '%s': %s", filename, e)

if files_moved == 0:
    logging.info("No images were found to move.")

logging.info("Image processing completed.")
