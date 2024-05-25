import csv
import json

# Function to convert JSON bounding box data to 'x1, y1, x2, y2' format
def convert_to_x1y1x2y2(json_data):
    
    print(len(json_data[0]))
    csv_data = []
    for bbox in json_data:
        x1 = bbox['x']
        y1 = bbox['y']
        x2 = bbox['x'] + bbox['width']
        y2 = bbox['y'] + bbox['height']
        label = str(bbox['labels'][0])
        csv_data.append([x1, y1, x2, y2, label])
    return csv_data

# Input and output file paths
input_file = 'input.csv'
output_file = 'output2_1im.csv'

# Read JSON data from input CSV file and write to output CSV file
with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)
    
    # Write header
    writer.writerow(['x1', 'y1', 'x2', 'y2'])
    
    for row in reader:
        # Parse JSON string to Python data structure
        json_data = json.loads(row['annotation_bboxes'])
        
        # Convert JSON data to 'x1, y1, x2, y2' format
        csv_data = convert_to_x1y1x2y2(json_data)
        
        # Write to output CSV file
        writer.writerows(csv_data)
        break
