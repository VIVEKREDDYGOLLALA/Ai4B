from PIL import Image, ImageDraw, ImageFont
import csv
from surya.ordering import batch_ordering
from surya.model.ordering.processor import load_processor
from surya.model.ordering.model import load_model
import matplotlib.pyplot as plt
import json
import uuid

def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        data = [list(map(float, row)) for row in reader]
    return data

def convert_to_tuples(order_predictions, width_im, height_im):
    tuples = []
    for order_result in order_predictions:
        boxes = []
        for order_box in order_result.bboxes:
            bbox = order_box.bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            # Generate unique alphanumeric ID
            box_id = str(uuid.uuid4())[:8]  # Using first 8 characters for simplicity
            boxes.append({
                'id': box_id,
                'bbox': [(bbox[0] * width_im) / 100, (bbox[1] * height_im) / 100, (bbox[2] * width_im) / 100,
                         (bbox[3] * height_im) / 100],
                'order': order_box.position
            })
        tuples.append(boxes)
    return tuples
def draw_and_save_image(image, order_predictions, output_image_path):
    draw = ImageDraw.Draw(image)
    for boxes in order_predictions:
        for prediction in boxes:
            bbox = prediction['bbox']
            draw.rectangle(bbox, outline="red")
            draw.text((bbox[0], bbox[1]), str(prediction['order']), fill="red")
    image.save(output_image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def generate_relations_json(order_predictions_tuples):
    relations = []
    for boxes in order_predictions_tuples:
        for i in range(len(boxes) - 1):
            relations.append({
                "type": "relation",
                "to_id": boxes[i+1]['id'],
                "labels": ["continues-to"],
                "from_id": boxes[i]['id'],
                "direction": "right"
            })
    return {"bboxes_relation_json": relations}

IMAGE_PATH = 'test1.png'
CSV_FILE_PATH = 'output_boxes_1_im.csv'
OUTPUT_IMAGE_PATH = 'output_image.png'
JSON_OUTPUT_PATH = 'output_relations.json'

# Open the image
image = Image.open(IMAGE_PATH)
width_im, height_im = image.size
print(f"Image dimensions: {width_im}, {height_im}")

# Read bounding box coordinates from CSV file
bboxes = read_csv(CSV_FILE_PATH)

# Load model and processor
model = load_model()
processor = load_processor()

# Perform batch ordering
order_predictions = batch_ordering([image], [bboxes], model, processor)

# Convert order_predictions to tuples for compatibility
order_predictions_tuples = convert_to_tuples(order_predictions, width_im, height_im)

# Draw bounding boxes and reading order on the image, then save and display
draw_and_save_image(image, order_predictions_tuples, OUTPUT_IMAGE_PATH)

# Generate the relations JSON and save to a file
relations_json = generate_relations_json(order_predictions_tuples)
with open(JSON_OUTPUT_PATH, 'w') as json_file:
    json.dump(relations_json, json_file, indent=2)

print(f"Relations JSON saved to {JSON_OUTPUT_PATH}")
