from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Load the image
# image = Image.open("/home/gaorz/Downloads/fox_ear_0lvl.png")
#
# # Define crop box (left, upper, right, lower)
# left = 250
# top = 200
# right = left + 200
# bottom = top + 200
# cropped_image = image.crop((left, top, right, bottom))
#
# # Convert to RGB to avoid PDF corruption
# cropped_image_rgb = cropped_image.convert("RGB")
#
# # Save the cropped image as PDF
# cropped_image_rgb.save("/home/gaorz/Downloads/fox_ear_0lvl_cropped.pdf", "PDF")
#
# # Show the cropped image
# plt.imshow(cropped_image)
# plt.axis('off')
# plt.show()


# Load the image
image = Image.open("/media/gaorz/b5df3483-c11a-42f1-b414-023f33bc5312/home/ruize/3d_vnn_ref/fox_front.png").convert("RGB")

# Define crop box (left, upper, right, lower)
left = 670
top = 60
right = left + 60
bottom = top + 60

# Draw a red rectangle on the original image
draw = ImageDraw.Draw(image)
# Rectangle outline width (optional)
outline_width = 3
for i in range(outline_width):  # To make thicker outline
    draw.rectangle(
        [left - i, top - i, right + i, bottom + i],
        outline="red"
    )

# Save the modified image with the red square as PNG (optional)
# image.save("/home/gaorz/Downloads/fox_ear_sdf_with_red_square.png")

# Save the modified image as PDF
image.save("/home/gaorz/Downloads/fox_ear_sdf_with_red_square.pdf", "PDF")

# Show the image with the red square
plt.imshow(image)
plt.axis('off')
plt.show()
