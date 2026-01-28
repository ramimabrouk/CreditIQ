import pandas as pd

# 1. Charger ton fichier
df = pd.read_csv('C:/Users/SAIF/.gemini/antigravity/scratch/credit_decision_memory/data/german_credit_data.csv')

# 2. Nettoyage rapide (Enlever la colonne index inutile si elle existe)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# 3. Création de la "Description Textuelle"
# C'est LA clé de ton projet. On mélange tout dans une phrase pour l'IA.
def creer_description(row):
    text = (
        f"Client de {row['Age']} ans, sexe {row['Sex']}. "
        f"Demande de prêt de {row['Credit amount']} DM pour {row['Purpose']}. "
        f"Durée: {row['Duration']} mois. "
        f"Logement: {row['Housing']}. "
        f"Compte épargne: {row['Saving accounts']}."
    )
    return text

# On applique cette fonction à toutes les lignes
df['Full_Description'] = df.apply(creer_description, axis=1)

# 4. On garde juste ce dont on a besoin : La description et le Risque
df_final = df[['Full_Description', 'Risk']]

# 5. Sauvegarde le résultat
df_final.to_csv('donnees_pretes_pour_vecteurs.csv', index=False)

print("✅ C'est bon ! Le fichier 'donnees_pretes_pour_vecteurs.csv' est créé.")
print("Exemple de ce que l'IA va lire :")
print(df_final['Full_Description'].iloc[0])