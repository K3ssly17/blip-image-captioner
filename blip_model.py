from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image


class BlipCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)

    def caption(self, image_path):

        image = Image.open(image_path).convert("RGB")


        inputs = self.processor(image, return_tensors="pt")

        output = self.model.generate(**inputs)


        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption


if __name__ == "__main__":
    blip = BlipCaptioner()
    result = blip.caption("test.jpg")
    print(result)
