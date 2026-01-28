from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pandas as pd
import numpy as np
import os
import shutil

# --- NETTOYAGE (Optionnel) ---
# Si le dossier existe déjà, on le supprime pour repartir à zéro
if os.path.exists("./ma_memoire_qdrant"):
    shutil.rmtree("./ma_memoire_qdrant")
    print("🧹 Ancienne mémoire effacée.")

print("1️⃣ Chargement des fichiers...")
try:
    df = pd.read_csv('base_de_connaissance.csv')
    vectors = np.load('mes_vecteurs.npy')
    print(f"   -> {len(df)} clients chargés.")
except FileNotFoundError:
    print("❌ ERREUR : Fichiers manquants (étape 2).")
    exit()

print("2️⃣ Initialisation de Qdrant...")
# On crée une base de données LOCALE (sur ton disque dur)
client = QdrantClient(path="./ma_memoire_qdrant")

collection_name = "dossiers_clients"

# Qdrant a besoin de savoir la taille des vecteurs (384 pour all-MiniLM-L6-v2)
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

print("3️⃣ Remplissage de la mémoire...")

points = []
# On prépare les données au format Qdrant ("PointStruct")
for idx, row in df.iterrows():
    # On crée un "Point" pour chaque client
    point = PointStruct(
        id=idx,  # ID unique (0, 1, 2...)
        vector=vectors[idx].tolist(), # Le vecteur mathématique
        payload={ # Les infos textuelles (Payload)
            "description": row['Full_Description'],
            "risk_label": row['Risk']
        }
    )
    points.append(point)

# On envoie tout d'un coup (upsert)
# Note : Pour des millions de lignes, on ferait des paquets (batchs), 
# mais pour 1000 lignes, ça passe d'un coup.
operation_info = client.upsert(
    collection_name=collection_name,
    points=points
)

print("\n✅ SUCCÈS ! La mémoire Qdrant est construite.")
print(f"📁 Données sauvegardées dans le dossier './ma_memoire_qdrant'")
print(f"   Statut de l'opération : {operation_info.status}")