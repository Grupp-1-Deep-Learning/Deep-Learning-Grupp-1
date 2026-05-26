import gradio as gr
import numpy as np
import joblib
from PIL import Image
from pathlib import Path
from datetime import datetime
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model


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
            elif model_path.suffix == ".keras":
                try:
                    loaded_models[model_path.name] = load_model(model_path)
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
        return None, "Rita ett tecken först 🙂", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

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
        return None, "Jag hittar inget ritat tecken 🙂", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

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
        "xgboost_lettermodel.json",'logistic_lettermodel.joblib',"cnn_lettermodel.keras"
    )

    if model_choice in letter_models:
        img_28 = img_28.rotate(90, expand=False)

    filename = SAVE_DIR / f"drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    img_28.save(filename)

    pixels = np.array(img_28).reshape(1, 784)

    if model_choice == "Alla modeller":
        prediction, o1, o2, o3 = predict_all_models(pixels)
    elif model_choice in loaded_models:
        prediction, o1, o2, o3 = predict_single_model(model_choice, pixels)
    else:
        prediction, o1, o2, o3 = "Ingen modell vald eller modellen hittades inte.", None, None, None

    if o1 is not None:
        return (
            img_28, 
            prediction, 
            gr.update(value=f"Välj {o1}", visible=True),
            gr.update(value=f"Välj {o2}", visible=True),
            gr.update(value=f"Välj {o3}", visible=True),
            gr.update(value="", visible=False)
        )
    else:
        return (
            img_28, 
            prediction, 
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False)
        )


def predict_single_model(model_name, pixels):
    model = loaded_models.get(model_name)
    if model is None:
        return f"Modellen {model_name} kunde inte laddas.", None, None, None
    
    # Speciell hantering för XGBoost-lettermodellen för att visa topp 3 gissningar, baserat på confidence 
    if model_name == "xgboost_lettermodel.json" and hasattr(model, "predict_proba"):
        probs = model.predict_proba(pixels)[0]
        top_3_indices = np.argsort(probs)[-3:][::-1]
        
        result_text = ""
        options = []
        for i in top_3_indices:
            confidence = probs[i] * 100
            display_prediction = chr(int(i) + 65)
            result_text += f"{model_name} gissar: {display_prediction}\nSäkerhet: {confidence:.1f}%\n\n"
            options.append(display_prediction)
            
        return result_text.strip(), options[0], options[1], options[2]
    
    if model_name.endswith(".keras"):
        cnn_pixels = pixels.reshape(1, 28, 28, 1) / 255.0
        probs = model.predict(cnn_pixels, verbose=0)[0]
        prediction = np.argmax(probs)
        confidence = probs[prediction] * 100
    else:
        prediction = model.predict(pixels)[0]
        confidence = None

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(pixels)[0]
            confidence = probs[int(prediction)] * 100
    
    if "letter" in model_name.lower():
        display_prediction = chr(int(prediction) + 65)
    else:
        display_prediction = str(prediction)

    if confidence is not None:
        return f"{model_name} gissar: {display_prediction}\nSäkerhet: {confidence:.1f}%", None, None, None

    return f"{model_name} gissar: {display_prediction}", None, None, None


def predict_all_models(pixels):
    if not loaded_models:
        return "Inga modeller är inladdade i systemet.", None, None, None
        
    results = []
    for model_name in loaded_models.keys():
        res = predict_single_model(model_name, pixels)
        results.append(res[0])

    return "\n\n".join(results), None, None, None


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
            
            # Topp 3 gissningar för bokstavsmodellen
            with gr.Row():
                btn_opt1 = gr.Button("Välj 1", visible=False)
                btn_opt2 = gr.Button("Välj 2", visible=False)
                btn_opt3 = gr.Button("Välj 3", visible=False)
                
            confirmation = gr.Textbox(label="Ditt val: ", visible=False)

    btn = gr.Button("Tolka tecken")

    sketchpad.clear( # Kallar på clear
        fn=reset_canvas,
        outputs=sketchpad
    )
    
    btn.click(
        fn=prepare_image, # fn står för "function" och anger vilken funktion som ska köras när knappen klickas
        inputs=[sketchpad, model_choice],
        outputs=[preview, result, btn_opt1, btn_opt2, btn_opt3, confirmation]
    )
    
    def confirm_choice(btn_text):
        letter = btn_text.replace("Välj ", "")
        return gr.update(value=f"Du har valt tecknet: {letter}", visible=True)

    # Lägger till click events för varje knapp som visar topp 3 gissningar, och kopplar dem till confirm_choice-funktionen
    btn_opt1.click(fn=confirm_choice, inputs=btn_opt1, outputs=confirmation)
    btn_opt2.click(fn=confirm_choice, inputs=btn_opt2, outputs=confirmation)
    btn_opt3.click(fn=confirm_choice, inputs=btn_opt3, outputs=confirmation)


if __name__ == "__main__":
    demo.launch()