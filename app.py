import gradio as gr
import numpy as np
import joblib
from PIL import Image
from pathlib import Path
from datetime import datetime

SAVE_DIR = Path("saved_drawings")
SAVE_DIR.mkdir(exist_ok=True)

RF_MNIST_MODEL_PATH = Path("trained_models/random_forest_mnist.joblib")
XGBOOST_MODEL_PATH = Path("trained_models/xgboost_mnist.joblib")
RF_EMNIST_MODEL_PATH = Path("trained_models/random_forest_emnist_balanced.joblib")

random_forest_mnist_model = joblib.load(RF_MNIST_MODEL_PATH)
xgboost_model = joblib.load(XGBOOST_MODEL_PATH)

emnist_bundle = joblib.load(RF_EMNIST_MODEL_PATH)
random_forest_emnist_model = emnist_bundle["model"]
emnist_label_map = emnist_bundle["label_map"]


def reset_canvas():
    return {
        "background": Image.new("L", (400, 400), 0),
        "layers": [],
        "composite": None
    }


def prepare_image(editor_value, model_choice):
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

    if model_choice == "Random Forest MNIST":
        prediction = predict_random_forest_mnist(pixels)
    elif model_choice == "Random Forest EMNIST":
        prediction = predict_random_forest_emnist(pixels)
    elif model_choice == "XGBoost MNIST":
        prediction = predict_xgboost(pixels)
    elif model_choice == "Alla modeller":
        prediction = predict_all_models(pixels)
    else:
        prediction = "Ingen modell vald."

    return img_28, prediction


def predict_random_forest_mnist(pixels):
    prediction = random_forest_mnist_model.predict(pixels)[0]

    if hasattr(random_forest_mnist_model, "predict_proba"):
        probs = random_forest_mnist_model.predict_proba(pixels)[0]
        confidence = probs[int(prediction)] * 100
        return f"Random Forest MNIST gissar: {prediction}\nSäkerhet: {confidence:.1f}%"

    return f"Random Forest MNIST gissar: {prediction}"


def predict_random_forest_emnist(pixels):
    prediction_label = random_forest_emnist_model.predict(pixels)[0]
    prediction_char = emnist_label_map[int(prediction_label)]

    if hasattr(random_forest_emnist_model, "predict_proba"):
        probs = random_forest_emnist_model.predict_proba(pixels)[0]

        class_index = list(random_forest_emnist_model.classes_).index(prediction_label)
        confidence = probs[class_index] * 100

        return (
            f"Random Forest EMNIST gissar: {prediction_char}\n"
            f"Label: {prediction_label}\n"
            f"Säkerhet: {confidence:.1f}%"
        )

    return f"Random Forest EMNIST gissar: {prediction_char}\nLabel: {prediction_label}"


def predict_xgboost(pixels):
    prediction = xgboost_model.predict(pixels)[0]

    if hasattr(xgboost_model, "predict_proba"):
        probs = xgboost_model.predict_proba(pixels)[0]
        confidence = probs[int(prediction)] * 100
        return f"XGBoost MNIST gissar: {prediction}\nSäkerhet: {confidence:.1f}%"

    return f"XGBoost MNIST gissar: {prediction}"


def predict_all_models(pixels):
    results = [
        predict_random_forest_mnist(pixels),
        predict_xgboost(pixels),
        predict_random_forest_emnist(pixels),
    ]

    return "\n\n".join(results)


with gr.Blocks(title="Teckenigenkänning") as demo:
    gr.Markdown("# Teckenigenkänning")
    gr.Markdown("Rita ett tecken i rutan och klicka på **Tolka tecken**.")

    with gr.Row():
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
                choices=[
                    "Random Forest MNIST",
                    "XGBoost MNIST",
                    "Random Forest EMNIST",
                    "Alla modeller"
                ],
                value="Random Forest MNIST",
                label="Välj modell"
            )

            result = gr.Textbox(
                label="Resultat från modell",
                lines=8
            )

    btn = gr.Button("Tolka tecken")

    sketchpad.clear(
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