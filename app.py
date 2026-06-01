import gradio as gr
import numpy as np
import joblib
import re
from PIL import Image
from pathlib import Path
from datetime import datetime
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model
from collections import Counter



SAVE_DIR = Path("saved_drawings")
SAVE_DIR.mkdir(exist_ok=True)


MODEL_ACCURACY = {
    "random_forest_mnist.joblib": 96.8,
    "xgboost_mnist.joblib": 97.0,
    "cnn_combined_model.keras": 98.5,
    "ann_model.keras": 97.5,
    "xgboost_lettermodel.json": 88.0,
    "logistic_lettermodel.joblib": 84.0,
    "cnn_lettermodel.keras": 92.0,
    "swe_chars_model.keras": 90.0,
}

LETTER_GROUP_BOOST = 1.35
DIGIT_GROUP_BOOST = 1.0
COMBINED_GROUP_BOOST = 1.0


# Kasta din modell i "trained_models" mappen, så laddas den automatiskt när appen startar.
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
                    model = load_model(model_path)
                    loaded_models[model_path.name] = model
                    # Warm-up för att förhindra fördröjning vid första klicket
                    if "cnn" in model_path.name or "swe_chars" in model_path.name:
                        model.predict(np.zeros((1, 28, 28, 1)), verbose=0)
                    elif "ann" in model_path.name:
                        model.predict(np.zeros((1, 784)), verbose=0)
                except Exception:
                    pass


def reset_canvas(): # Den här funktionen nollställer canvas, behövs för att vi ska börja med penseln. 
    return {
        "background": Image.new("L", (340, 340), 0), # Skapar en svart bakgrund
        "layers": [],
        "composite": None
    }

def prepare_image(editor_value, model_choice):
    """
    Tar Bildern från Gradio ImageEditor,
    beskär bort tom yta,
    centrerar tecknet,
    gör om till 28x28 = 784 pixlar,
    sparar bilden och skickar vidare till vald modell.
    """

    if editor_value is None or editor_value.get("composite") is None:
        return None, "Rita ett tecken först 🙂", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

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
        return None, "Jag hittar inget ritat tecken 🙂", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

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

    real_model_name = display_to_model.get(model_choice)
    letter_models = (
        "xgboost_lettermodel.json",
        'logistic_lettermodel.joblib',
        "cnn_lettermodel.keras",
    )


    filename = SAVE_DIR / f"drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    img_28.save(filename)

    img_28_digit = img_28.copy()

    img_28_letter = img_28.copy()
    img_28_letter = img_28_letter.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    img_28_letter = img_28_letter.rotate(90, expand=False)

    model_choice = real_model_name

    if model_choice == "Alla modeller":
        prediction, o1, o2, o3 = predict_all_models(img_28_digit, img_28_letter)

    elif model_choice in loaded_models:
        if model_choice in letter_models:
            pixels = np.array(img_28_letter).reshape(1, 784)
            preview_img = img_28_letter
        else:
            pixels = np.array(img_28_digit).reshape(1, 784)
            preview_img = img_28_digit

        prediction, o1, o2, o3 = predict_single_model(model_choice, pixels)
        img_28 = preview_img
    else:
        prediction, o1, o2, o3 = "Ingen modell vald eller modellen hittades inte.", None, None, None

    if o1 is not None and o1 != "":
        return (
            img_28, 
            prediction, 
            gr.update(value=f"Välj {o1}", visible=True),
            gr.update(value=f"Välj {o2}" if o2 else "", visible=bool(o2)),
            gr.update(value=f"Välj {o3}" if o3 else "", visible=bool(o3))
        )
    else:
        return (
            img_28, 
            prediction, 
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
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

        # CNN model
        if model_name == "cnn_combined_model.keras" or "cnn" in model_name or "swe_chars" in model_name:
            model_pixels = pixels.reshape(1, 28, 28, 1) / 255.0

        # ANN model
        elif model_name == "ann_model.keras" or "ann" in model_name:
            model_pixels = pixels.reshape(1, 784)

        else:
            model_pixels = pixels

        probs = model.predict(model_pixels, verbose=0)[0]
        
        # Snabbt spår för swe_chars_model.keras
        if "swe_chars" in model_name:
            swe_mapping = {0: 'Å', 1: 'Ä', 2: 'Ö', 3: 'å', 4: 'ä', 5: 'ö', 6: 'null'}
            mapping_to_use = swe_mapping

            if mapping_to_use:
                best_idx = np.argmax(probs)
                best_char = mapping_to_use.get(int(best_idx), str(best_idx))
                
                if best_char.lower() == "null":
                    conf = probs[best_idx] * 100
                    return f"{model_name} gissar: null\nSäkerhet: {conf:.1f}%", None, None, None
                
                top_indices = np.argsort(probs)[::-1]
                valid_options = []
                result_text = ""
                
                for idx in top_indices:
                    char = mapping_to_use.get(int(idx), str(idx))
                    if char.lower() != "null":
                        conf = probs[idx] * 100
                        if idx == best_idx or conf >= 20.0:
                            result_text += f"{model_name} gissar: {char}\nSäkerhet: {conf:.1f}%\n\n"
                            valid_options.append(char)
                
                while len(valid_options) < 3:
                    valid_options.append(None)
                    
                return result_text.strip(), valid_options[0], valid_options[1], valid_options[2]
            else:
                prediction = np.argmax(probs)
                confidence = probs[prediction] * 100
                display_prediction = str(prediction)
                return f"{model_name} gissar: {display_prediction}\nSäkerhet: {confidence:.1f}%", None, None, None

        # Speciell hantering för CNN Combined och ANN för att visa topp 3 gissningar
        if model_name == "cnn_combined_model.keras" or model_name == "ann_model.keras" or "cnn" in model_name:
            top_3_indices = np.argsort(probs)[-3:][::-1]
            
            result_text = ""
            options = []
            for i in top_3_indices:
                confidence = probs[i] * 100
                
                if "digit" in model_name:
                    display_prediction = str(i)
                elif "letter" in model_name:
                    display_prediction = chr(int(i) + 65)
                elif model_name == "ann_model.keras":
                    if i <= 9:
                        display_prediction = str(i)
                    elif i <= 35:
                        display_prediction = chr(i - 10 + ord("A"))
                    else:  # 36-61 → lowercase
                        display_prediction = chr(i - 36 + ord("a"))
                else:
                    if i <= 9:
                        display_prediction = str(i)
                    else:
                        display_prediction = chr(i - 10 + 65)
                
                result_text += f"{model_name} gissar: {display_prediction}\nSäkerhet: {confidence:.1f}%\n\n"
                options.append(display_prediction)
                
            while len(options) < 3:
                options.append("")
                
            return result_text.strip(), options[0], options[1], options[2]

        prediction = np.argmax(probs)
        confidence = probs[prediction] * 100

    else:
        prediction = model.predict(pixels)[0]
        confidence = None

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(pixels)[0]
            confidence = probs[int(prediction)] * 100  


    if model_name == "cnn_combined_model.keras":
        if prediction <= 9:
            display_prediction = str(prediction)
        else:
            display_prediction = chr(prediction - 10 + 65)
    elif model_name == "ann_model.keras":
        if prediction <= 9:
            display_prediction = str(prediction)
        elif prediction <= 35:
            display_prediction = chr(prediction - 10 + ord("A"))
        else:
            display_prediction = chr(prediction - 36 + ord("a"))
    elif "letter" in model_name.lower():
        display_prediction = chr(int(prediction) + 65)
    else:
        display_prediction = str(prediction)

    if confidence is not None:
        return f"{model_name} gissar: {display_prediction}\nSäkerhet: {confidence:.1f}%", None, None, None

    return f"{model_name} gissar: {display_prediction}", None, None, None


def extract_prediction(result_text):
    """
    Plockar ut första gissningen från text som:
    'modellnamn gissar: A'
    """
    match = re.search(r"gissar:\s*(\S+)", result_text)
    if match:
        return match.group(1)
    return None

def is_digit_prediction(prediction):
    return prediction.isdigit()


def is_letter_prediction(prediction):
    return prediction.isalpha() and prediction.lower() != "null"


def is_digit_model(model_name):
    return "digit" in model_name or "mnist" in model_name


def is_letter_model(model_name):
    return "letter" in model_name


def get_model_group(model_name):
    if is_letter_model(model_name):
        return "letter"
    if is_digit_model(model_name):
        return "digit"
    return "combined"

def predict_all_models(img_28_digit, img_28_letter):
    if not loaded_models:
        return "Inga modeller är inladdade i systemet.", None, None, None

    # Ifall swe_chars har en träffm då ska den vara exklusiv
    if "swe_chars_model.keras" in loaded_models:
        pixels = np.array(img_28_digit).reshape(1, 784)
        res_txt, o1, o2, o3 = predict_single_model("swe_chars_model.keras", pixels)
        pred = extract_prediction(res_txt)
        if pred and pred.lower() != "null":
            summary = "🧠 Sammanvägt resultat\n"
            summary += f"Slutlig gissning (Swe Chars): {pred}\n\n"
            summary += "--- Endast Swe Chars Modell ---\n\n"
            summary += res_txt
            return summary, o1, o2, o3

    all_texts = []
    votes = []
    weighted_scores = {}

    letter_models = (
        "xgboost_lettermodel.json",
        "logistic_lettermodel.joblib",
        "cnn_lettermodel.keras",
    )

    for model_name in loaded_models.keys():

        if model_name in letter_models:
            pixels = np.array(img_28_letter).reshape(1, 784)
        else:
            pixels = np.array(img_28_digit).reshape(1, 784)

        result_text, _, _, _ = predict_single_model(model_name, pixels)

        all_texts.append(result_text)

        prediction = extract_prediction(result_text)

        if prediction is not None and prediction.lower() != "null":
            group = get_model_group(model_name)

            accuracy = MODEL_ACCURACY.get(model_name, 90)

            if group == "letter":
                group_size = sum(1 for name in loaded_models if get_model_group(name) == "letter")
            elif group == "digit":
                group_size = sum(1 for name in loaded_models if get_model_group(name) == "digit")
            else:
                group_size = 1

            group_size = max(group_size, 1)

            if group == "letter":
                group_boost = LETTER_GROUP_BOOST
            elif group == "digit":
                group_boost = DIGIT_GROUP_BOOST
            else:
                group_boost = COMBINED_GROUP_BOOST

            normalized_score = (accuracy / group_size) * group_boost

            votes.append(prediction)

            weighted_scores[prediction] = (
                weighted_scores.get(prediction, 0) + normalized_score
            )

    if not votes:
        return "\n\n".join(all_texts), None, None, None

    vote_count = Counter(votes)

    digit_votes = [v for v in votes if is_digit_prediction(v)]
    letter_votes = [v for v in votes if is_letter_prediction(v)]

    digit_count = Counter(digit_votes)
    letter_count = Counter(letter_votes)

    best_digit = digit_count.most_common(1)[0][0] if digit_count else None
    best_letter = letter_count.most_common(1)[0][0] if letter_count else None


    # Vid lika röster -> högst accuracy-vikt
    final_prediction = max(
        weighted_scores,
        key=weighted_scores.get
    )

    summary = "🧠 Sammanvägt resultat\n"
    summary += f"Slutlig gissning: {final_prediction}\n"
    summary += f"Viktad score: {weighted_scores[final_prediction]:.1f}\n\n"

    if best_letter:
        summary += f"Bästa bokstav: {best_letter} ({letter_count[best_letter]} röster)\n"

    if best_digit:
        summary += f"Bästa siffra: {best_digit} ({digit_count[best_digit]} röster)\n"

    summary += "\n📊 Alla röster:\n"

    for prediction, count in vote_count.most_common():
        summary += (
            f"{prediction}: "
            f"{count} röst(er), "
            f"score {weighted_scores[prediction]:.1f}\n"
        )

    summary += "\n--- Alla modeller ---\n\n"
    summary += "\n\n".join(all_texts)

    sorted_options = sorted(weighted_scores.keys(), key=lambda x: weighted_scores[x], reverse=True)
    opt1 = sorted_options[0] if len(sorted_options) > 0 else None
    opt2 = sorted_options[1] if len(sorted_options) > 1 else None
    opt3 = sorted_options[2] if len(sorted_options) > 2 else None

    return summary, opt1, opt2, opt3

# Mapping mellan snyggt namn och riktigt filnamn
display_to_model = {}

for model_name in loaded_models.keys():

    clean_name = (
        model_name
        .replace(".joblib", "")
        .replace(".json", "")
        .replace(".keras", "")
        .replace("_", " ")
        .title()
    )

    display_to_model[clean_name] = model_name

display_to_model["Alla modeller"] = "Alla modeller"

model_choices = list(display_to_model.keys())

if len(model_choices) > 0:
    default_value = model_choices[0]
else:
    default_value = "Alla modeller"

with gr.Blocks(
    title="Teckenigenkänning"
) as demo:

    with gr.Column(elem_id="app-wrapper"):

        gr.HTML("""
        <div class="main-title">
            ✍️ Teckenigenkänning
        </div>

        <div class="subtitle">
            Rita ett tecken och låt modellen försöka tolka det.
        </div>
        """)

        with gr.Row(equal_height=True, elem_classes="main-row"):

            with gr.Column(elem_classes=["app-panel", "draw-column"]):
                sketchpad = gr.ImageEditor(
                    label="Rita tecken här",
                    type="pil",
                    image_mode="L",
                    sources=(),
                    interactive=True,
                    brush=gr.Brush(
                        colors=["#FFFFFF"],
                        default_color="#FFFFFF",
                        color_mode="fixed",
                        default_size=10
                    ),
                    eraser=gr.Eraser(default_size=20),
                    height=320,
                    width=320,
                    canvas_size=(320, 320),
                    layers=False,
                    value=reset_canvas()
                )

            with gr.Column(elem_classes=["middle-column"]):

                with gr.Column(elem_classes=["app-panel", "model-column"]):
                    model_choice = gr.Radio(
                        choices=model_choices,
                        value=default_value,
                        label="Välj modell",
                        interactive=True,
                        elem_classes="model-radio"
                    )

                btn = gr.Button(
                    "🔍 Tolka tecken",
                    elem_classes="primary-btn"
                )

                with gr.Row(elem_classes="choice-buttons"):
                    btn_opt1 = gr.Button("Välj 1", visible=False)
                    btn_opt2 = gr.Button("Välj 2", visible=False)
                    btn_opt3 = gr.Button("Välj 3", visible=False)

            with gr.Column(elem_classes=["app-panel", "result-column"]):
                text_input = gr.Textbox(
                    label="Sparade tecken", 
                    interactive=True
                )
                
                preview = gr.Image(
                    label="Sparad 28x28-bild",
                    height=120,
                    type="pil"
                )

                result = gr.Textbox(
                    label="Resultat från modell",
                    lines=7,
                    elem_classes="result-box"
                )

    sketchpad.clear( # Kallar på clear
        fn=reset_canvas,
        outputs=sketchpad
    )
    
    # Kör prepare_image och kedjar sedan en .then() för att garantera UI-uppdatering och synkronisering på skärmen direkt vid första klicket
    btn.click(
        fn=prepare_image, # fn står för "function" och anger vilken funktion som ska köras när knappen klickas
        inputs=[sketchpad, model_choice],
        outputs=[preview, result, btn_opt1, btn_opt2, btn_opt3]
    ).then(
        fn=lambda: (gr.update(), gr.update(), gr.update()),
        inputs=None,
        outputs=[btn_opt1, btn_opt2, btn_opt3]
    )
    
    def append_choice(current_text, btn_text):
        if current_text is None:
            current_text = ""
        letter = btn_text.replace("Välj ", "")
        return current_text + letter

    # Lägger till click events för varje knapp som visar gissningarna, så att de kan sparas i textinput
    btn_opt1.click(fn=append_choice, inputs=[text_input, btn_opt1], outputs=text_input)
    btn_opt2.click(fn=append_choice, inputs=[text_input, btn_opt2], outputs=text_input)
    btn_opt3.click(fn=append_choice, inputs=[text_input, btn_opt3], outputs=text_input)


if __name__ == "__main__":
    demo.launch(css_paths="style.css")