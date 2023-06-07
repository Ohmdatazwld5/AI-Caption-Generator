from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
from flask import Flask, request, render_template

app = Flask(__name__)

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
# Define number of captions to generate per image
captions_per_img = 3

# Update gen_kwargs with num_return_sequences
gen_kwargs.update({"num_return_sequences": captions_per_img})

def generate_captions(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds_list = []

    for idx in range(len(output_ids)):
        preds = tokenizer.batch_decode(output_ids[idx], skip_special_tokens=True)
        preds_cleaned = [pred.strip() for pred in preds]
        preds_list.append(preds_cleaned)

    return preds_list

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get uploaded files
        uploaded_files = request.files.getlist('file[]')

        # Get file paths from uploaded files 
        image_paths = []
        for file in uploaded_files:
            file.save(file.filename)
            image_paths.append(file.filename)

        captions_set_for_all_images = generate_captions(image_paths)

        # Return the result
        return render_template('result.html', captions_set=captions_set_for_all_images)

if __name__ == '__main__':
    app.run(debug=True)