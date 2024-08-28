
from inference import get_model
from PIL import Image, ImageDraw
model = get_model(model_id="mavic2-ecipz/1")
image = Image.open("11.jpg")
draw = ImageDraw.Draw(image)
results = model.infer("11.jpg")
print(results)
model.predict(image, confidence=40, overlap=30).save("prediction.jpg")
# example box object from the Pillow library
for bounding_box in results:
    x1 = bounding_box['x'] - bounding_box['width'] / 2
    x2 = bounding_box['x'] + bounding_box['width'] / 2
    y1 = bounding_box['y'] - bounding_box['height'] / 2
    y2 = bounding_box['y'] + bounding_box['height'] / 2
    box = (x1, x2, y1, y2)

    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # Save or display the modified image

    image.show()  # Display the image
# from roboflow import Roboflow
# rf = Roboflow(api_key="8ezpQ7uQn7RqISDkKsnM")
# from inference import get_model

# model = get_model(model_id="mavic2-ecipz/1")


# # infer on a local image
# print(model.predict("11.jpg", confidence=40, overlap=30).json())

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())