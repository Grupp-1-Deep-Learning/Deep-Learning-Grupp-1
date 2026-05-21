import gradio as gr
import numpy as np
import joblib
from PIL import Image
from pathlib import Path
from datetime import datetime
from xgboost import XGBClassifier


SAVE_DIR = Path("saved_drawings")
SAVE_DIR.mkdir(exist_ok=True)

# Kasta din modell i "trained_models"-mappen så laddas den automatiskt när appen startar.
MODELS_DIR = Path("trained_models") 
loaded_models = {}

if MODELS_DIR.exists() and MODELS_DIR.is_dir():
    for model_path in MODELS_DIR.iterdir():
        if model_path.is_file():
            if model_path.suffix == ".joblib":
                try:
                    loaded_models[model_path.name] = joblib.load(model_path)
                except Exception:
                    pass
            elif model_path.suffix == ".json":
                try:
                    model = XGBClassifier()
                    model.load_model(model_path)
                    loaded_models[model_path.name] = model
                except Exception:
                    pass


def reset_canvas(): # Den här funktionen nollställer canvas, behövs för att vi ska börja med penseln. 
    return {
        "background": Image.new("L", (400, 400), 0), # Skapar en svart bakgrund
        "layers": [],
        "composite": None
    }

def prepare_image(editor_value, model_choice):
    """
    Tar bilden från Gradio ImageEditor,
    beskär bort tom yta,
    centrerar tecknet,
    gör om till 28x28 = 784 pixlar,
    sparar bilden och skickar vidare till vald modell.
    """

    if editor_value is None or editor_value.get("composite") is None:
        return None, "Rita ett tecken först 🙂"

    img = editor_value["composite"]

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img = img.convert("L")

    # Gör till numpy-array
    arr = np.array(img)

    # Om bilden har svart tecken på vit bakgrund, invertera
    # Målet är: svart bakgrund, vitt tecken
    if arr.mean() > 127:
        arr = 255 - arr

    # Ta bort svaga pixlar/brus
    arr[arr < 30] = 0

    # Hitta alla pixlar där något är ritat
    coords = np.argwhere(arr > 0)

    if coords.size == 0:
        return None, "Jag hittar inget ritat tecken 🙂"

    # Bounding box runt tecknet
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    cropped = arr[y0:y1 + 1, x0:x1 + 1]

    # Gör bilden kvadratisk
    h, w = cropped.shape
    size = max(h, w)

    square = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    square[y_offset:y_offset + h, x_offset:x_offset + w] = cropped

    square_img = Image.fromarray(square)

    # Skala till ca 20x20 så det finns marginal runt tecknet
    square_img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Lägg centrerat på 28x28 canvas
    img_28 = Image.new("L", (28, 28), 0)

    x = (28 - square_img.width) // 2
    y = (28 - square_img.height) // 2

    img_28.paste(square_img, (x, y))

    letter_models = (
        "xgboost_lettermodel.json",
    )

    if model_choice in letter_models:
        img_28 = img_28.rotate(90, expand=False)

    filename = SAVE_DIR / f"drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    img_28.save(filename)

    pixels = np.array(img_28).reshape(1, 784)

    if model_choice == "Alla modeller":
        prediction = predict_all_models(pixels)
    elif model_choice in loaded_models:
        prediction = predict_single_model(model_choice, pixels)
    else:
        prediction = "Ingen modell vald eller modellen hittades inte."

    return img_28, prediction
# Fyll på med fler predict-funktioner här när vi lägger till fler modeller.


def predict_single_model(model_name, pixels):
    model = loaded_models.get(model_name)
    if model is None:
        return f"Modellen {model_name} kunde inte laddas."

    prediction = model.predict(pixels)[0]
    
    if "letter" in model_name.lower():
        display_prediction = chr(int(prediction) + 65)
    else:
        display_prediction = str(prediction)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(pixels)[0]
        confidence = probs[int(prediction)] * 100
        return f"{model_name} gissar: {display_prediction}\nSäkerhet: {confidence:.1f}%"

    return f"{model_name} gissar: {display_prediction}"


def predict_all_models(pixels):
    if not loaded_models:
        return "Inga modeller är inladdade i systemet."
        
    results = []
    for model_name in loaded_models.keys():
        results.append(predict_single_model(model_name, pixels))

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
                default_size=10
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

            model_choices = []

            # Hämta ut alla filnamn (nycklar) på de modeller som lyckades laddas in från mappen
            for model_name in loaded_models.keys():
                model_choices.append(model_name)

            model_choices.append("Alla modeller")

            # Standardval som ska visas i rullgardinsmenyn när appen startar
            if len(loaded_models) > 0:
                all_model_names = list(loaded_models.keys())
                default_value = all_model_names[0]
            else:
                # Om mappen var tom och inga modeller hittades
                default_value = "Alla modeller"

            # Med den nya koden är vi mindre begränsade av våra modellval
            model_choice = gr.Dropdown(
                choices=model_choices,
                value=default_value,
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