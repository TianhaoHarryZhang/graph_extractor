import pandas as pd 
import numpy as np 
from PIL import Image
import anthropic
import os
import io
import json
import base64
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_raw_graph_data_from_images(image, key) -> dict: #image: PIL.Image.Image
        
        client = anthropic.Anthropic(api_key=key)

        # Resize the image if it's too large
        #max_size = (1600, 1600)  # Adjust these dimensions as needed
        #image.thumbnail(max_size, Image.LANCZOS)
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        if len(img_byte_arr) > 5 * 1024 * 1024:  # 5MB in bytes
            #print("Image size exceeds 5MB. Skipping this image.")
            raise Exception ("Image size exceeds 5MB. Skipping this image.")

        # Encode the binary data to base64
        img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

        #print(image, img_base64)

        full_response = ""
        chunk_size = 4096  # Adjust this value based on your needs

        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,  # Increased max_tokens
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": """Extract all 2D plots from this image as structured JSON data. 
                            For each recognized plot, extract all data needed to reconstruct it, including, graph title, x/y axis labels and data points.

Format the extracted data as a dictionary with the following structure:
{
    "title": "The full title of the graph",
    "x-label": "label for x-axis",
    "y-label": "label for y-axis",

    "data": {

    	"plot-1 label":[[x1,x2,x3,x4...x500],[y1,y2,y3,y4...y500]],
    	"plot-2 label":[[x1,x2,x3,x4...x500],[y1,y2,y3,y4...y500]],
    	"plot-3 label":[[x1,x2,x3,x4...x500],[y1,y2,y3,y4...y500]],
    	...

    }
}"""
                        }
                    ]
                }
            ]
        )
        #print(message.content)
        
        # Collect the full response
        for chunk in tqdm(message.content[0].text):
            full_response += chunk

        if len(full_response) >= chunk_size:
                print(f"Claude model received {len(full_response)} characters")    
        
        #self.raw_api_response = full_response #for debugging purpose

        
        try:
            raw_graph_data = json.loads(full_response)

        except json.JSONDecodeError as e:
            print("Failed to parse JSON due to json decoding error. Skipping this image.")
            print("Raw response: ",full_response)
            raise e



        return raw_graph_data


def reconstruct(graph_data):

    # Create the plot
    plt.figure(figsize=(8, 6))  # Set the figure size

    data = graph_data["data"]

    for label, values in data.items():

        x = values[0]
        y = values[1]

        plt.plot(x, y, label=label)

    # Add labels and title
    plt.title(graph_data["title"], fontsize=16)
    plt.xlabel(graph_data["x-label"], fontsize=14)
    plt.ylabel(graph_data["y-label"], fontsize=14)

    
    plt.legend(fontsize=12)

    # Show the plot
    plt.show()


if __name__ == '__main__':


    image_path = "graph.png"

    image = Image.open(image_path)
    
    graph_data = get_raw_graph_data_from_images(image=image, key=os.environ.get('ANTHROPIC_API_KEY'))

    reconstruct(graph_data)
	

	




