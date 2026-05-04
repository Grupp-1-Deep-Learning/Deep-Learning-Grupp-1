import gradio as gr
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime

SAVE_DIR = Path("saved_drawings")
SAVE_DIR.mkdir(exist_ok=True)


def prepare_image(editor_value):
    """
    Tar bilden från Gradio ImageEditor,
    gör om den till 28x28 = 784 pixlar,
    sparar bilden och skickar vidare till dummy-backend.
    """

    if editor_value is None or editor_value.get("composite") is None:
        return None, "Rita ett tecken först 🙂"

    img = editor_value["composite"]

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    # Gör till gråskala
    img = img.convert("L")

    # Skala till 28x28 = 784 pixlar
    img_28 = img.resize((28, 28))

    # Spara bilden
    filename = SAVE_DIR / f"drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    img_28.save(filename)

    # Gör om till numpy-array för framtida modell
    pixels = np.array(img_28).reshape(1, 784)

    # Dummy-svar tills modellerna finns
    prediction = dummy_backend_predict(pixels)

    return img_28, prediction


def dummy_backend_predict(pixels):
    """
    Här byter vi senare ut mot riktig backend/model-anrop.
    """
    return "Modellen gissar något här... (byt ut mot riktig gissning senare)"


with gr.Blocks(title="Teckenigenkänning") as demo:
    gr.Markdown("# Teckenigenkänning")
    gr.Markdown("Rita ett tecken i rutan och klicka på **Tolka tecken**.")

    with gr.Row():
        sketchpad = gr.ImageEditor(
            label="Rita tecken här",
            type="pil",
            image_mode="L",
            brush=gr.Brush(colors=["#000000"], default_size=20),
            eraser=gr.Eraser(default_size=20),
            height=400,
            width=400,
        )

        with gr.Column():
            preview = gr.Image(label="Sparad 28x28-bild")
            result = gr.Textbox(label="Resultat från modell")

    btn = gr.Button("Tolka tecken")
    btn.click(
        fn=prepare_image,
        inputs=sketchpad,
        outputs=[preview, result]
    )


if __name__ == "__main__":
    demo.launch()