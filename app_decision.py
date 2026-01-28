from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np

# --- CONFIGURATION ---
SEUIL_ANOMALIE = 0.25  # Si la similarité est sous 0.25, c'est louche (Anomaly)

print("1️⃣ Chargement du cerveau et de la mémoire...")
model = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient(path="./ma_memoire_qdrant")

# --- SIMULATION D'UN NOUVEAU CLIENT ---
# Tu pourras changer ce texte pour tester d'autres profils !
nouveau_dossier = "Client de 22 ans, femme. Demande 5000 DM pour une voiture. Pas d'épargne. Travaille depuis 1 an."

print(f"\n📄 Analyse du dossier : \"{nouveau_dossier}\"")

# 2. Vectorisation du nouveau client
vecteur_client = model.encode(nouveau_dossier)

# 3. Recherche dans la mémoire (Qdrant)
print("🔍 Recherche de cas similaires dans le passé...")
resultats = client.search(
    collection_name="dossiers_clients",
    query_vector=vecteur_client,
    limit=5  # On regarde les 5 plus proches voisins (K-Nearest Neighbors)
)

# 4. Analyse des résultats (L'Intelligence du système)
votes_risques = []
scores_similarite = []

print("\n--- CAS SIMILAIRES TROUVÉS ---")
for hit in resultats:
    score = hit.score
    risque = hit.payload['risk_label']
    description = hit.payload['description']
    
    scores_similarite.append(score)
    votes_risques.append(risque)
    
    # On affiche les détails pour l'explicabilité (Audit-Ready)
    print(f"   👉 Similarité: {score:.2f} | Risque Passé: {risque} | {description[:80]}...")

# 5. Prise de décision (Logique métier)
nb_bad = votes_risques.count("bad")
nb_good = votes_risques.count("good")
moyenne_similarite = sum(scores_similarite) / len(scores_similarite)

print("\n--- 🤖 DÉCISION DU SYSTÈME ---")

# DÉTECTION D'ANOMALIE
if moyenne_similarite < SEUIL_ANOMALIE:
    print("⚠️ ALERTE ANOMALIE : Ce profil ne ressemble à rien de connu !")
    print(f"   (Score moyen de similarité trop faible : {moyenne_similarite:.2f})")
    print("   -> Action : Revue manuelle obligatoire.")

# DÉCISION DE CRÉDIT
elif nb_bad > nb_good:
    print("❌ RECOMMANDATION : REFUSER LE PRÊT")
    print(f"   Raison : {nb_bad} des 5 profils similaires ont fait défaut.")
else:
    print("✅ RECOMMANDATION : ACCORDER LE PRÊT")
    print(f"   Raison : {nb_good} des 5 profils similaires ont remboursé sans problème.")
    print(f"   Confiance : {moyenne_similarite:.2f}/1.0")