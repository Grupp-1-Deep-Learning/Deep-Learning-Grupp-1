# model.py
import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models

def load_char_images(base_dir):
    """
    Denna funktion letar efter mappar som heter t.ex. 'ao_upper' (för Å) 
    och laddar in alla bilder den hittar där.
    Den ger varje bild en siffra (label) mellan 0 och 5.
    0 = Å, 1 = Ä, 2 = Ö, 3 = å, 4 = ä, 5 = ö.
    """
    images = []
    labels = []
    # Här bestämmer vi vilken mapp som hör till vilken siffra (klass/label)
    folder_mapping = {
        'ao_upper': 0, 'ae_upper': 1, 'oe_upper': 2,
        'ao_lower': 3, 'ae_lower': 4, 'oe_lower': 5
    }
    
    # Gå igenom varje mapp och siffra i listan ovan
    for folder_name, label in folder_mapping.items():
        folder_path = os.path.join(base_dir, folder_name) # Skapa hela sökvägen till mappen
        if os.path.exists(folder_path): # Kolla om mappen faktiskt finns
            # Leta upp alla filer i mappen (*.* betyder alla filnamn och alla filändelser)
            for file_path in glob.glob(os.path.join(folder_path, "*.*")):
                try:
                    # Öppna bilden, gör den gråskalig ('L') och ändra storlek till 28x28 pixlar
                    img = Image.open(file_path).convert('L').resize((28, 28))
                    
                    # Gör om bilden till en array av siffror och dela med 255.0 
                    # så alla pixlar får ett värde mellan 0.0 och 1.0 (detta kallas normalisering)
                    images.append(np.array(img, dtype=np.float32) / 255.0)
                    
                    # Spara vilken siffra (Å, Ä, Ö osv) denna bild tillhör
                    labels.append(label)
                except:
                    # Om bilden är trasig eller något går fel, strunta i den och gå vidare (pass)
                    pass
    
    # Returnera bilderna och deras labels som numpy-arrays, vilket AI-modeller vill ha
    return np.array(images), np.array(labels)

def generate_noise_images(num_images=2000):
    """
    Skapar fejkade/slumpmässiga bilder (brus).
    Detta hjälper modellen att förstå vad som absolut INTE är ett Å, Ä eller Ö.
    Den ritar bara ut slumpmässiga gråa/vita rutor på en svart bakgrund.
    """
    images = []
    for _ in range(num_images):
        # Skapa en helt svart bild (28x28 pixlar med värdet 0)
        img = np.zeros((28, 28), dtype=np.float32)
        
        # Rita ut 1 till 4 slumpmässiga "block" på den svarta bilden
        for _ in range(np.random.randint(1, 5)):
            x, y = np.random.randint(0, 20, 2) # Slumpa en startpunkt (x,y)
            w, h = np.random.randint(2, 10, 2) # Slumpa en bredd (w) och höjd (h)
            # Fyll blocket med en slumpmässig grå färg (mellan 0.5 och 1.0)
            img[y:y+h, x:x+w] = np.random.uniform(0.5, 1.0)
            
        images.append(img)
    return np.array(images)

def main():
    """
    Huvudfunktionen där allt sätts ihop och modellen tränas.
    """
    # Hitta var på datorn detta skript (model.py) ligger
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Skapa en mapp som heter 'models' där vi ska spara den färdiga modellen
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True) # Skapar mappen om den inte redan finns
    
    # 1. LADDA IN DINA EGNA TECKEN (ÅÄÖåäö)
    # x_chars är själva bilderna, y_chars är siffran (0-5) som talar om vilken bokstav det är
    x_chars, y_chars = load_char_images(current_dir)
    
    if len(x_chars) == 0:
        print("Hittade inga bilder att träna på. Skapa mappar och lägg in bilder först.")
        return # Avbryt om det inte finns några bilder att träna på

    # 2. LADDA IN DATA SOM MODELLEN SKA IGNORERA (Klass 6 / 'null')
    # a) Ladda in 5000 vanliga siffror (0-9) från MNIST-datasetet
    (x_mnist, _), _ = tf.keras.datasets.mnist.load_data()
    x_mnist = x_mnist[:5000].astype("float32") / 255.0 # Normalisera till 0.0 - 1.0
    
    # b) Skapa 2000 slumpmässiga "brus"-bilder med funktionen vi skrev ovan
    x_noise = generate_noise_images(2000)
    
    # c) Kolla om du har sparat några egna "felaktiga" tecken i en mapp som heter 'ignore'
    ignore_path = os.path.join(current_dir, "ignore")
    ignore_images = []
    if os.path.exists(ignore_path):
        for file_path in glob.glob(os.path.join(ignore_path, "*.*")):
            try:
                img = Image.open(file_path).convert('L').resize((28, 28))
                ignore_images.append(np.array(img, dtype=np.float32) / 255.0)
            except:
                pass
                
    # Slå ihop alla "felaktiga" bilder (siffror och brus till att börja med)
    x_neg_list = [x_mnist, x_noise]
    
    # Om du hade egna 'ignore'-bilder, kopiera dem många gånger så modellen verkligen lär sig dem
    if len(ignore_images) > 0:
        ignore_images = np.array(ignore_images)
        # Räkna ut hur många gånger de behöver kopieras (max 1000 kopior totalt)
        repeat_ignore = max(1, 1000 // len(ignore_images)) 
        ignore_repeated = np.repeat(ignore_images, repeat_ignore, axis=0)
        x_neg_list.append(ignore_repeated) # Lägg till i listan över "felaktiga" bilder
        
    # Slå ihop listan `x_neg_list` till en stor array med alla "negativa" bilder
    x_neg = np.concatenate(x_neg_list, axis=0)
    
    # Ge alla dessa "felaktiga" bilder siffran (label) 6
    # Klass 0-5 var ju åäö, så klass 6 betyder "Inget av dem / Ignore"
    y_neg = np.ones(len(x_neg)) * 6 
    
    # 3. BALANSERA DATAN
    # AI-modeller tränar bäst om de har lika många bilder av varje sak.
    # Eftersom vi har jättemånga "ignore"-bilder (över 7000), måste vi kopiera
    # upp dina ÅÄÖ-bilder så att de blir ungefär lika många.
    
    x_pos = np.expand_dims(x_chars, -1) # Keras vill ha formen (28, 28, 1) inte bara (28, 28)
    repeat_factor = max(1, len(x_neg) // len(x_chars)) # Räkna ut hur mycket vi behöver kopiera
    x_pos = np.repeat(x_pos, repeat_factor, axis=0)    # Kopiera bilderna
    y_pos = np.repeat(y_chars, repeat_factor, axis=0)  # Kopiera siffrorna/labels
    
    x_neg = np.expand_dims(x_neg, -1) # Se till att de negativa bilderna också har formen (28, 28, 1)
    
    # Slå ihop alla "positiva" (ÅÄÖ) och "negativa" (ignore) till den slutgiltiga träningsdatan
    x_train = np.concatenate((x_pos, x_neg), axis=0)
    y_train = np.concatenate((y_pos, y_neg), axis=0)

    # 4. BYGG SJÄLVA AI-MODELLEN
    
    # Data Augmentation: Slumpar lite rotation och inzoomning under träningen
    # Detta gör att modellen inte blir förvirrad om du ritar ett Ä lite snett
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.02),
        layers.RandomTranslation(0.02, 0.02),
        layers.RandomZoom(0.02)
    ])

    # Skapa ett Convolutional Neural Network (CNN)
    # Ett CNN är bäst på att analysera bilder och hitta mönster (som prickar och streck)
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)), # Förväntar sig inkommande bilder på 28x28 pixlar i svartvitt (1 färgkanal)
        data_augmentation,               # Skaka om bilden lite (rotera/zooma)
        
        # Första "bildfiltret" (Convolutional layer). Letar efter enkla streck och kanter.
        layers.Conv2D(64, (3, 3), activation='relu'), 
        layers.MaxPooling2D((2, 2)), # Förminskar bilden för att bara spara den viktigaste informationen
        
        # Andra "bildfiltret". Kombinerar streck för att hitta svårare former (cirklar, hörn).
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(), # Platta till bilden från ett rutnät till en lång rad med siffror
        
        # En "vanlig" hjärn-del (Dense layer) som kollar på alla upphittade mönster
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3), # Glömmer slumpmässigt 30% av vad den lärt sig (förhindrar att den "memorerar" data utan att förstå)
        
        # Utgången (Output layer). Har 7 valmöjligheter (0, 1, 2, 3, 4, 5 eller 6).
        # Softmax gör om gissningarna till procent (så alla 7 alternativ tillsammans blir 100%).
        layers.Dense(7, activation='softmax') 
    ])

    # Inställningar för hur den ska lära sig
    # 'adam' är motorn, 'sparse_categorical_crossentropy' är sättet den mäter hur fel den har
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # 5. STARTA TRÄNINGEN
    # epochs=15 betyder att den tittar igenom alla bilder 15 gånger för att bli bättre och bättre
    # batch_size=32 betyder att den tränar på 32 bilder åt gången innan den uppdaterar hjärnan
    model.fit(x_train, y_train, epochs=15, batch_size=32)
    
    # Spara den färdigtränade modellen på hårddisken
    model_path = os.path.join(models_dir, "swe_chars_model.keras")
    model.save(model_path)
    print("Modellen är tränad och sparad som 'swe_chars_model.keras'")

if __name__ == "__main__":
    main()