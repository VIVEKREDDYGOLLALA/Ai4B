import csv
from PIL import Image, ImageDraw
from surya.ordering import batch_ordering
from surya.model.ordering.processor import load_processor
from surya.model.ordering.model import load_model

# Define the paths to your CSV file
CSV_PATH = "test.csv"

# Load the CSV data
image_paths = ['test1.png']
annotation_data = []

with open(CSV_PATH, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        image_paths.append(row['image_path'])
        annotation_data.append(row['annotation_data'])

# Initialize lists to store image and annotation pairs
image_annotations = []

# Load images and annotations
for image_path, annotation_path in zip(image_paths, annotation_data):
    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Load JSON data (assuming annotation_data contains JSON strings)
    annotation_bboxes = json.loads(annotation_path)

    # Convert boxes into the required format [x1, y1, x2, y2]
    bboxes = []
    for bbox_info in annotation_bboxes:
        try:
            x1 = bbox_info['x']
            y1 = bbox_info['y']
            x2 = x1 + bbox_info['width']
            y2 = y1 + bbox_info['height']
            bboxes.append([x1, y1, x2, y2])
        except KeyError as e:
            print(f"Error: Key '{e}' not found in one of the bounding box dictionaries.")
            continue
    
    # Append image and annotations to the list
    image_annotations.append((image, draw, bboxes))

# Load the model and processor
model = load_model()
processor = load_processor()

# Process each image with its annotations
for image, draw, bboxes in image_annotations:
    # Get the reading order predictions
    order_predictions = batch_ordering([image], [bboxes], model, processor)

    # Draw bounding boxes
    for bbox in bboxes:
        draw.rectangle(bbox, outline="red")

    # Print the order predictions
    for order in order_predictions[0]:
        try:
            print(annotation_bboxes[order]['labels'])
        except IndexError as e:
            print(f"Error: Index out of range while accessing order {order}.")

    # Save or show the image
    image.show()
