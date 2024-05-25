from PIL import Image, ImageDraw, ImageFont
import csv
from surya.ordering import batch_ordering
from surya.model.ordering.processor import load_processor
from surya.model.ordering.model import load_model
import matplotlib.pyplot as plt

def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        print(type(reader))
        next(reader)  # Skip the header row
        data = [list(map(float, row)) for row in reader]
    return data

def convert_to_tuples(order_predictions):
    tuples = []
    for order_result in order_predictions:
        boxes = []
        for order_box in order_result.bboxes:
            bbox = order_box.bbox
            width = bbox[2]-bbox[0]
            height = bbox[3] - bbox[1]
            print(bbox)
            boxes.append({
                'bbox': [(bbox[0] * width_im)/100, (bbox[1]*height_im)/100, (bbox[2]*width_im)/100, (bbox[3]*height_im)/100],
                'order': order_box.position
            })
        tuples.append(boxes)
    return tuples

def draw_and_save_image(image, order_predictions, output_image_path):
    draw = ImageDraw.Draw(image)
    for boxes in order_predictions:
        for prediction in boxes:
            bbox = prediction['bbox']
            print(prediction)
            draw.rectangle(bbox, outline="red")
            # draw.text((bbox[0], bbox[1] - 20), label, fill="red")  # Print label above the bbox
            draw.text((bbox[0], bbox[1]), str(prediction['order']), fill="red")
    image.save(output_image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

IMAGE_PATH = 'test1.png'
# CSV_FILE_PATH = 'output_boxes_1_im.csv'
CSV_FILE_PATH = 'output2_1im.csv'
OUTPUT_IMAGE_PATH = 'output_image.png'

# Open the image
image = Image.open(IMAGE_PATH)
width_im, height_im = image.size
print(f"Image dimensions: {width_im}, {height_im}")

# Read bounding box coordinates from CSV file
bboxes = read_csv(CSV_FILE_PATH)
# print(bboxes)

# Load model and processor
model = load_model()
processor = load_processor()

# Perform batch ordering
order_predictions = batch_ordering([image], [bboxes], model, processor)
print(order_predictions)
# Convert order_predictions to tuples for compatibility
order_predictions_tuples = convert_to_tuples(order_predictions)
# print(order_predictions_tuples)

# Draw bounding boxes and reading order on the image, then save and display
draw_and_save_image(image, order_predictions_tuples, OUTPUT_IMAGE_PATH, )
