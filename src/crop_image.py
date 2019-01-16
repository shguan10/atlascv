from PIL import Image
import os

def crop_image(original, name):
  final_width = 672
  final_height = 384
  width, height = original.size   # Get dimensions
  left = (width - final_width) / 2
  top = (height - final_height) / 2
  right = (width + final_width) / 2
  bottom = (height + final_height) / 2
  cropped_example = original.crop((left, top, right, bottom))
  cropped_example.save(name[:-4] + "_crop.png", "PNG")

rootdir = "data_road/training/gt_image_2"
for subdir, dirs, files in os.walk(rootdir):
  for file in files:
    #print os.path.join(subdir, file)
    filepath = subdir + os.sep + file
    if not filepath.endswith("_crop.png"):
      file_name = filepath
      test_image = file_name
      original = Image.open(test_image)

      crop_image(original, filepath)
    print(filepath)