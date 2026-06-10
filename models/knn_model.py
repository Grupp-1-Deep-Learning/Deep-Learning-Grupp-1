import os 
import glob
import numpy as np 
from PIL import Image 
from sklearn.neighbors import KNeighborsClassifier 
import joblib 
def load_images(base_dir):
    # Skapa tomma listor där vi kan spara pixlarna från varje bild (images) och vilken siffra den föreställer (labels)
    images = []
    labels = []
    
    # Loopa igenom siffrorna 0 till 9, eftersom vi förväntar oss mappar döpta till just "0", "1", osv.
    for label in range(10):
        # Bygg ihop den fullständiga sökvägen till mappen för den aktuella siffran
        folder_path = os.path.join(base_dir, str(label))
        
        # Kontrollera om denna mapp faktiskt existerar på hårddisken
        if os.path.exists(folder_path):
            # glob.glob letar upp alla filer ("*.*") inuti denna specifika mapp
            for file_path in glob.glob(os.path.join(folder_path, "*.*")):
                try:
                    # 1. Öppna bildfilen
                    # 2. .convert('L') gör om bilden till gråskala (1 färgkanal istället för 3 för RGB)
                    # 3. .resize((28, 28)) skalar om bilden till exakt 28x28 pixlar oavsett originalstorlek
                    img = Image.open(file_path).convert('L').resize((28, 28))
                    
                    # Gör om bilden till en Numpy-array av flyttal (decimaltal).
                    # .flatten() är avgörande här: KNN kan inte läsa en 2D-bild (28x28), 
                    # så vi plattar ut den till en lång 1D-lista med 784 pixlar (28 * 28 = 784).
                    # Vi delar sedan med 255.0 för att "normalisera" värdena så att de ligger mellan 0.0 (svart) och 1.0 (vitt).
                    images.append(np.array(img, dtype=np.float32).flatten() / 255.0)
                    
                    # Lägg till mappens namn (siffran 0-9) som facit för just denna bild
                    labels.append(label)
                except:
                    # Om filen inte är en bild (t.ex. en dold systemfil) går vi bara vidare och ignorerar den
                    pass
                    
    # Returnera datan som Numpy-arrayer, vilket är formatet som scikit-learn kräver
    return np.array(images), np.array(labels)

def main():
    # Hämta den exakta mappen där detta Python-skript (model.py) just nu befinner sig
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Definiera en undermapp som heter "models" där vi vill spara vår färdiga modell
    models_dir = os.path.join(current_dir, "models")
    
    # Skapa denna 'models'-mapp. exist_ok=True gör att programmet inte kraschar om mappen redan finns
    os.makedirs(models_dir, exist_ok=True)
    
    # Anropa funktionen för att ladda in alla bilder och deras etiketter från 0-9 mapparna
    x_train, y_train = load_images(current_dir)
    
    # Kontrollera säkerhetsmässigt om listan med bilder är tom (inga bilder hittades)
    if len(x_train) == 0:
        # Om vi inte har någon data, skriv ut ett felmeddelande och avbryt skriptet
        print("Hittade inga bilder att träna på. Skapa mappar 0-9 och lägg in bilder först.")
        return

    # Sätt hur många grannar (K) som KNN-algoritmen ska titta på när den gissar.
    # Normalt använder vi 3, men om det finns färre än 3 bilder totalt i träningsdatan, 
    # så sänker vi K till det totala antalet bilder för att undvika felmeddelanden.
    n_neighbors = min(3, len(x_train))
    
    # Skapa en helt ny, otränad KNN-modell med inställningarna ovan
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Träna modellen! Detta säger åt modellen att lära sig kopplingen mellan bilderna (x_train) och rätt siffra (y_train)
    model.fit(x_train, y_train)
    
    # Definiera vad den färdiga filen ska heta och var den ska sparas
    model_path = os.path.join(models_dir, "knn_model.joblib")
    
    # Använd joblib för att skriva (dump) ner den tränade modellen till hårddisken
    joblib.dump(model, model_path)
    
    # Berätta för användaren att processen är klar och exakt var filen hamnade
    print(f"Modellen är tränad och sparad som '{model_path}'")

# Detta block ser till att koden inuti main() bara körs om vi startar filen direkt (t.ex. 'python model.py').
# Det förhindrar att träningen startar automatiskt om vi importerar funktioner härifrån till en annan fil.
if __name__ == "__main__":
    main()