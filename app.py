import gradio as gr
import numpy as np
import joblib
from PIL import Image
from pathlib import Path
from datetime import datetime

SAVE_DIR = Path("saved_drawings")
SAVE_DIR.mkdir(exist_ok=True)

RF_MODEL_PATH = Path("trained_models/random_forest_mnist.joblib")
random_forest_model = joblib.load(RF_MODEL_PATH)



def reset_canvas(): # Den här funktionen nollställer canvas, behövs för att vi ska börja med penseln. 
    return {
        "background": Image.new("L", (400, 400), 0), # Skapar en svart bakgrund
        "layers": [],
        "composite": None
    }

def prepare_image(editor_value, model_choice):
    """
    Tar bilden från Gradio ImageEditor,
    gör om den till 28x28 = 784 pixlar,
    sparar bilden och skickar vidare till vald modell.
    """

    if editor_value is None or editor_value.get("composite") is None:
        return None, "Rita ett tecken först 🙂"

    img = editor_value["composite"]

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img = img.convert("L")
    img_28 = img.resize((28, 28))

    filename = SAVE_DIR / f"drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    img_28.save(filename)

    pixels = np.array(img_28).reshape(1, 784)

    if model_choice == "Random Forest":
        prediction = predict_random_forest(pixels)
    elif model_choice == "Alla modeller":
        prediction = predict_all_models(pixels)
    else:
        prediction = "Ingen modell vald."

    return img_28, prediction


def predict_random_forest(pixels):
    prediction = random_forest_model.predict(pixels)[0]

    if hasattr(random_forest_model, "predict_proba"):
        probs = random_forest_model.predict_proba(pixels)[0]
        confidence = probs[int(prediction)] * 100
        return f"Random Forest gissar: {prediction}\nSäkerhet: {confidence:.1f}%"

    return f"Random Forest gissar: {prediction}"


def predict_all_models(pixels):
    results = []

    results.append(predict_random_forest(pixels))

    # Lägg till fler modeller här senare:
    # results.append(predict_svm(pixels))
    # results.append(predict_cnn(pixels))

    return "\n\n".join(results)


with gr.Blocks(title="Teckenigenkänning") as demo:
    gr.Markdown("# Teckenigenkänning")
    gr.Markdown("Rita ett tecken i rutan och klicka på **Tolka tecken**.")

    with gr.Row(): # Lagt till så att vi börjar med penseln direkt
        sketchpad = gr.ImageEditor(
            label="Rita tecken här",
            type="pil",
            image_mode="L",
            sources=(),
            interactive=True,
            brush=gr.Brush( # Ställer in penseln
                colors=["#FFFFFF"],
                default_color="#FFFFFF",
                color_mode="fixed",
                default_size=20
            ),
            eraser=gr.Eraser(default_size=20),
            height=400,
            width=400,
            canvas_size=(400, 400),
            layers=False,
            value=reset_canvas()
        )

        with gr.Column():
            preview = gr.Image(
                label="Sparad 28x28-bild",
                height=80,
                type="pil"
            )

            model_choice = gr.Dropdown(
                choices=["Random Forest", "Alla modeller"],
                value="Random Forest",
                label="Välj modell"
            )
            
            result = gr.Textbox(
                label="Resultat från modell"
        )

    btn = gr.Button("Tolka tecken")

    sketchpad.clear( # Kallar på clear
        fn=reset_canvas,
        outputs=sketchpad
    )
    
    btn.click(
        fn=prepare_image,
        inputs=[sketchpad, model_choice],
        outputs=[preview, result]
    )


if __name__ == "__main__":
    demo.launch()