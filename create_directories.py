import os
import shutil
import logging

# Configure logging
logging.basicConfig(filename='move_images.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Source and target directories
source_dir = 'classifier'
target_dir_train = 'training_set'
target_dir_test = 'testing_set'

if not os.path.exists(source_dir):
    logging.error("Source directory '%s' does not exist.", source_dir)
    exit(1)

if not os.path.exists(target_dir_train):
    os.makedirs(target_dir_train)
if not os.path.exists(target_dir_test):
    os.makedirs(target_dir_test)

files_moved = 0

for filename in os.listdir(source_dir):
    if filename.endswith('.jpg'):
        source_file = os.path.join(source_dir, filename)
        # Add your logic here to decide whether to move the file to train or test
        # For example, move all files to training_set
        target_file = os.path.join(target_dir_train, filename)
        try:
            shutil.move(source_file, target_file)
            files_moved += 1
            logging.info("Moved file '%s' to '%s'.", filename, target_file)
        except Exception as e:
            logging.error("Error moving file '%s': %s", filename, e)

if files_moved == 0:
    logging.info("No images were found to move.")
