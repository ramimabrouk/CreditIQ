from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import sys

# 1. Chargement
print("1️⃣  Chargement des données...")
try:
    df = pd.read_csv('donnees_pretes_pour_vecteurs.csv')
    print(f"   -> Trouvé {len(df)} clients dans le fichier.")
except FileNotFoundError:
    print("❌ ERREUR : Le fichier 'donnees_pretes_pour_vecteurs.csv' est introuvable.")
    sys.exit()

# 2. Préparation du modèle IA
print("2️⃣  Chargement du cerveau (Modèle)...")
# La première fois, ça prendra un peu de temps pour télécharger le modèle (80Mo)
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Vectorisation
print("3️⃣  Traduction des textes en mathématiques (Patience)...")
descriptions = df['Full_Description'].tolist()
vectors = model.encode(descriptions)

# 4. Sauvegarde
print("4️⃣  Sauvegarde des résultats...")
# On sauvegarde les vecteurs dans un fichier .npy (format rapide)
np.save('mes_vecteurs.npy', vectors)
# On garde aussi les données textuelles liées
df.to_csv('base_de_connaissance.csv', index=False)

print("\n✅ SUCCÈS TOTAL !")
print(f"Tu as créé {vectors.shape[0]} vecteurs de dimension {vectors.shape[1]}.")
print("Fichiers créés : 'mes_vecteurs.npy' et 'base_de_connaissance.csv'")