MODELE_LLM = "qwen2.5:7b"                   # expérience possible : "qwen2.5:14b" (plus lent, plus fiable)
MODELE_EMBEDDING = "bge-m3"                 # même modèle pour indexer et pour interroger
MODELE_RERANKER = "BAAI/bge-reranker-v2-m3"

MODE_RECHERCHE = "dense"      # "bm25", "dense", "hybride" ou "rerank" : choisi d'après la mesure (section 13)
TOP_K = 4                     # passages donnés au LLM
CANDIDATS_RERANK = 12         # candidats relus par le reranker

TAUX_REMPLISSAGE = 0.90       # remplissage cible d'un contenant avant rotation
MAX_ROTATIONS_SEMAINE = 2     # au-delà, on ajoute un contenant
SUPPLEMENT_PRIORITE = 0.08    # option rotation 24 h (+8 % sur la rotation)
OBJECTIF_VALORISATION = 0.85  # exigence de l'article 6 du CCTP

ARTICLES_A_ANALYSER = [f"ART-{n:02d}" for n in range(2, 14)]   # articles porteurs d'exigences
NON_TROUVE = "Je n'ai pas trouvé cette information dans les documents."
TEMPS = {}                    # durée de chaque étape, pour le bilan

import re
from pathlib import Path

def decouper_markdown(chemin, corpus):
    texte = Path(chemin).read_text(encoding="utf-8")
    titre_doc = re.search(r"^# (.+)$", texte, re.M).group(1).strip()
    morceaux = []
    for n, partie in enumerate(re.split(r"^## ", texte, flags=re.M)[1:], 1):
        entete, _, corps = partie.partition("\n")
        entete, corps = entete.strip(), corps.strip()
        if corpus == "dce":
            identifiant = f"ART-{int(re.match(r'Article (\d+)', entete).group(1)):02d}"
        else:
            identifiant = f"B{Path(chemin).name[:2]}-{n:02d}"
        morceaux.append({
            "id": identifiant, "corpus": corpus, "document": titre_doc, "section": entete,
            "texte": corps, "texte_index": f"{titre_doc} — {entete}\n{corps}",
        })
    return morceaux

CHUNKS = []
for corpus in ["dce", "base"]:
    for chemin in sorted(Path(f"data/{corpus}").glob("*.md")):
        CHUNKS += decouper_markdown(chemin, corpus)
PAR_ID = {c["id"]: c for c in CHUNKS}

print(f"{len(CHUNKS)} morceaux : {sum(c['corpus'] == 'dce' for c in CHUNKS)} articles du CCTP, "
      f"{sum(c['corpus'] == 'base' for c in CHUNKS)} sections de la base interne")
print(PAR_ID["ART-06"]["texte_index"][:300])

import unicodedata
import numpy as np

STOPWORDS = set("""le la les l un une des du de d et ou a au aux en dans sur pour par avec sans ce cet cette ces
qui que quoi quel quelle quels quelles est sont etre il elle ils elles on se sa son ses leur leurs ne pas plus y t s n
qu c j m comment combien quand pourquoi doit doivent nous notre nos vous votre faut peut tout tous toute""".split())

def plier(texte):
    """minuscules + suppression des accents : 'Été' -> 'ete'"""
    texte = unicodedata.normalize("NFD", str(texte).lower())
    return "".join(c for c in texte if unicodedata.category(c) != "Mn")

def normaliser(texte):
    mots = [m for m in re.findall(r"\w+", plier(texte)) if m not in STOPWORDS]
    return [m[:-1] if len(m) > 3 and m[-1] in "sx" else m for m in mots]

def scores_bm25(requete, docs_tokens, k1=1.5, b=0.75):
    N = len(docs_tokens)
    avgdl = max(sum(len(d) for d in docs_tokens) / max(N, 1), 1)
    scores = np.zeros(N)
    for terme in set(normaliser(requete)):
        df = sum(terme in d for d in docs_tokens)
        if df == 0:
            continue
        idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
        for i, d in enumerate(docs_tokens):
            f = d.count(terme)
            if f:
                scores[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avgdl))
    return scores

def fusion_rrf(*classements, k=60):
    scores = {}
    for classement in classements:
        for rang, i in enumerate(classement):
            scores[i] = scores.get(i, 0) + 1 / (k + rang + 1)
    return sorted(scores, key=scores.get, reverse=True)

TOKENS = [normaliser(c["texte_index"]) for c in CHUNKS]
print(normaliser("Les pénalités de retard des rotations"))

import math
import pandas as pd

def lire_gisement(article=None):
    """Lit le tableau markdown de l'article 4 du CCTP."""
    lignes = (article or PAR_ID["ART-04"]["texte"]).splitlines()
    rangees = [[x.strip() for x in l.strip().strip("|").split("|")] for l in lignes if l.strip().startswith("|")]
    rangees = [r for r in rangees[1:] if not set("".join(r)) <= set("-: ")]   # retire en-tête et séparateur
    return pd.DataFrame([{"flux": r[0], "code": r[1], "tonnage_t": float(r[2].replace(",", ".")),
                          "dangereux": r[3].lower() == "oui"} for r in rangees])

GRILLE = pd.read_csv("data/base/grille_dimensionnement.csv")
GISEMENT = lire_gisement()

def calculer_dimensionnement(gisement, grille=GRILLE, priorite=False):
    d = gisement.merge(grille.drop(columns="dangereux"), on="flux", how="left", validate="1:1")
    if d["contenant"].isna().any():
        raise ValueError(f"Flux absents de la grille : {d.loc[d['contenant'].isna(), 'flux'].tolist()}")
    d["rotations_an"] = [math.ceil(t / (cu * TAUX_REMPLISSAGE)) for t, cu in zip(d.tonnage_t, d.charge_utile_t)]
    d["rotations_semaine"] = (d.rotations_an / 52).round(2)
    d["contenants"] = [max(1, math.ceil(r / 52 / MAX_ROTATIONS_SEMAINE)) for r in d.rotations_an]
    prix_rotation = d.rotation_eur * (1 + SUPPLEMENT_PRIORITE if priorite else 1)
    d["cout_location"] = d.location_mois_eur * 12 * d.contenants
    d["cout_transport"] = (d.rotations_an * prix_rotation).round(0)
    d["cout_traitement"] = d.tonnage_t * d.traitement_eur_t
    d["budget_annuel"] = d.cout_location + d.cout_transport + d.cout_traitement
    d["tonnes_valorisees"] = d.tonnage_t * d.taux_valorisation
    return d

def synthese(d):
    nd = d[~d.dangereux]
    taux = nd.tonnes_valorisees.sum() / nd.tonnage_t.sum()
    return {"tonnage_total_t": d.tonnage_t.sum(), "rotations_an": int(d.rotations_an.sum()),
            "contenants": int(d.contenants.sum()), "budget_annuel_eur": round(d.budget_annuel.sum()),
            "taux_valorisation_non_dangereux": round(taux, 4),
            "objectif_85_respecte": bool(taux >= OBJECTIF_VALORISATION)}

DIM = calculer_dimensionnement(GISEMENT)
DIM_PRIORITE = calculer_dimensionnement(GISEMENT, priorite=True)
COLONNES_DIM = ["flux", "tonnage_t", "contenant", "contenants", "rotations_an", "rotations_semaine",
                "budget_annuel", "taux_valorisation"]
display(DIM[COLONNES_DIM])
print("Standard (48 h) :", synthese(DIM))
print("Option 24 h     :", synthese(DIM_PRIORITE))

# Tests unitaires : le calcul doit rester juste quand on modifie le code
assert len(GISEMENT) == 9 and GISEMENT.tonnage_t.sum() == 2339
assert DIM.set_index("flux").loc["Métaux ferreux", "rotations_an"] == 91        # 650 / (8 × 0,9) = 90,3 -> 91
assert DIM.set_index("flux").loc["Sables de fonderie usagés", "rotations_an"] == 72
assert DIM.set_index("flux").loc["Métaux ferreux", "cout_traitement"] < 0      # reprise = recette
assert synthese(DIM)["objectif_85_respecte"]
assert synthese(DIM_PRIORITE)["budget_annuel_eur"] > synthese(DIM)["budget_annuel_eur"]
print("Tests du dimensionnement : OK ✅")