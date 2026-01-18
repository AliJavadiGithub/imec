import json

def find_max_id(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # This part handles the specific nested structure in your snippet
        # If your data is a list of objects, we extract the 'id' from each
        ids = []
        
        # Helper to recursively find all 'id' keys if the structure is deeply nested
        def extract_ids(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == 'id':
                        ids.append(v)
                    else:
                        extract_ids(v)
            elif isinstance(obj, list):
                for item in obj:
                    extract_ids(item)

        extract_ids(data)

        if not ids:
            return "No IDs found in the file."
        
        return max(ids)

    except FileNotFoundError:
        return "File not found."
    except json.JSONDecodeError:
        return "Invalid JSON format."

# Usage
file_name = 'tracking_results.json' # Replace with your filename
max_id = find_max_id(file_name)
print(f"The maximum ID is: {max_id}")