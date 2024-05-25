from PIL import Image, ImageDraw
import csv
from surya.ordering import batch_ordering
from surya.model.ordering.processor import load_processor
from surya.model.ordering.model import load_model
import matplotlib.pyplot as plt
import uuid
import json
import subprocess
import os

def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        data = [list(map(float, row)) for row in reader]
    return data

def read_labels_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        labels = [row[0] for row in reader]  # Assuming labels are in the first column
    return labels

def convert_to_tuples(order_predictions, width_im, height_im, labels, header_footer_labels):
    tuples = []
    header_footer_ids = {}
    label_index = 0
    
    for order_result in order_predictions:
        boxes = []
        for order_box in order_result.bboxes:
            bbox = order_box.bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            # Generate unique alphanumeric ID
            box_id = str(uuid.uuid4())[:8]  # Using first 8 characters for simplicity
            
            # Get the corresponding label
            label = labels[label_index]
            label_index += 1
            
            # Check if the label is header or footer
            if label in header_footer_labels:
                if label not in header_footer_ids:
                    header_footer_ids[label] = box_id  # Assign a unique ID for this header/footer label
                box_id = header_footer_ids[label]
                
            boxes.append({
                'id': box_id,
                'bbox': [(bbox[0] * width_im) / 100, (bbox[1] * height_im) / 100,
                         (bbox[2] * width_im) / 100, (bbox[3] * height_im) / 100],
                'order': order_box.position,
                'label': label
            })
        tuples.append(boxes)
    return tuples

def draw_and_save_image(image, order_predictions, labels, output_image_path):
    draw = ImageDraw.Draw(image)
    label_index = 0  # Index to keep track of the current label
    for boxes in order_predictions:
        for prediction in boxes:
            bbox = prediction['bbox']
            draw.rectangle(bbox, outline="red")
            draw.text((bbox[0], bbox[1]), str(prediction['order']), fill="red")
            draw.text((bbox[0], bbox[1] - 10), labels[label_index], fill="green")  # Print label above the bbox
            label_index = (label_index + 1) % len(labels)  # Move to the next label, looping back to the start if necessary
    image.save(output_image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def crop_and_run_ocr(image_path, boxes):
    for box in boxes:
        box_id = box['id']
        bbox = box['bbox']
        box_image_path = f'box_{box_id}.png'
        
        # Crop the box from the original image using OpenCV
        x1, y1, x2, y2 = bbox
        box_image = image.crop((x1, y1, x2, y2))
        box_image.save(box_image_path)
        
        # Run surya_ocr command on the cropped box image
        subprocess.run(['surya_ocr', box_image_path, '--images', '--langs', 'hi,en'])
        
        # Remove the temporary box image
        os.remove(box_image_path)

def generate_relations_json(order_predictions_tuples):
    relations = []
    for boxes in order_predictions_tuples:
        for i in range(len(boxes) - 1):
            relations.append({
                "type": "relation",
                "to_id": boxes[i + 1]['id'],
                "labels": ["continues-to"],
                "from_id": boxes[i]['id'],
                "direction": "right"
            })
    return {"bboxes_relation_json": relations}

IMAGE_PATH = 'test1.png'
CSV_FILE_PATH = 'output_boxes_1_im.csv'
LABELS_CSV_FILE_PATH = 'output1_1lab.csv'
OUTPUT_IMAGE_PATH = 'output_image.png'
JSON_OUTPUT_PATH = 'output_relations.json'

# Define header and footer labels
HEADER_FOOTER_LABELS = {'header', 'footer'}

# Open the image
image = Image.open(IMAGE_PATH)
width_im, height_im = image.size

# Read bounding box coordinates from CSV file
bboxes = read_csv(CSV_FILE_PATH)

# Load model and processor
model = load_model()
processor = load_processor()

# Perform batch ordering
order_predictions = batch_ordering([image], [bboxes], model, processor)

# Read labels from CSV file
labels = read_labels_csv(LABELS_CSV_FILE_PATH)

# Convert order_predictions to tuples for compatibility, considering header and footer
order_predictions_tuples = convert_to_tuples(order_predictions, width_im, height_im, labels, HEADER_FOOTER_LABELS)

# Draw bounding boxes and reading order on the image, then save and display
draw_and_save_image(image, order_predictions_tuples, labels, OUTPUT_IMAGE_PATH)

# Generate the relations JSON and save to a file
relations_json = generate_relations_json(order_predictions_tuples)
with open(JSON_OUTPUT_PATH, 'w') as json_file:
    json.dump(relations_json, json_file, indent=2)

# Draw bounding boxes and reading order on the image, then save and display
draw_and_save_image(image, order_predictions_tuples, labels, OUTPUT_IMAGE_PATH)

# Crop each box and run OCR
crop_and_run_ocr(image, order_predictions_tuples[0])  # Assuming there is only one page

print(f"Relations JSON saved to {JSON_OUTPUT_PATH}")
