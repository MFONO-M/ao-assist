"""Exécute TOUT le code du notebook hors Colab, en remplaçant les modèles (Ollama, reranker, LLM,
agent) par des simulations simples. Objectif : vérifier l'enchaînement, les DataFrames, l'export
Excel, le document Word et l'évaluation — pas la qualité des modèles."""
import hashlib, re, sys, types, os
from pathlib import Path
import numpy as np

ICI = Path(__file__).parent
os.chdir(sys.argv[1])  # dossier de travail contenant data/

# --- ollama.embed : sac de mots haché ---------------------------------------
def _vec(t):
    v = np.zeros(256)
    for m in re.findall(r"\w+", t.lower()):
        v[int(hashlib.md5(m.encode()).hexdigest(), 16) % 256] += 1
    return v + 1e-6
ollama = types.ModuleType("ollama")
ollama.embed = lambda model, input: {"embeddings": [_vec(t).tolist() for t in input]}
sys.modules["ollama"] = ollama

# --- CrossEncoder ------------------------------------------------------------
st = types.ModuleType("sentence_transformers")
class CrossEncoder:
    def __init__(self, *a, **k): pass
    def predict(self, paires):
        return [len(set(q.lower().split()) & set(p.lower().split())) for q, p in paires]
st.CrossEncoder = CrossEncoder
sys.modules["sentence_transformers"] = st

# --- LLM simulé ----------------------------------------------------------------
class Msg:
    def __init__(self, content, type_="ai"): self.content, self.type = content, type_
    def pretty_print(self): print(f"[{self.type}] {self.content[:120]}")

class FauxStructure:
    def __init__(self, schema): self.schema = schema
    def invoke(self, messages):
        contenu = messages[-1][1]
        nom = self.schema.__name__
        if nom == "ListeExigences":
            phrases = [p.strip() for p in re.split(r"(?<=[.;])\s+", contenu.split("\n", 1)[-1]) if len(p) > 30]
            Ex = self.schema.model_fields["exigences"].annotation.__args__[0]
            return self.schema(exigences=[Ex(intitule=p[:40], description=p, categorie="Délais",
                                             niveau="souhaité" if "appréci" in p else "obligatoire",
                                             valeur_cible="") for p in phrases[:4]])
        if nom == "Evaluation":
            ids = re.findall(r"\[(B\d{2}-\d{2})\]", contenu)
            return self.schema(statut="Partiel" if "24" in contenu.split("PASSAGES")[0] else "Conforme",
                               justification="simulation", reponse_proposee=f"Nous répondons [{ids[0]}].",
                               sources=ids[:1])
        raise ValueError(nom)

class ChatOllama:
    def __init__(self, **k): pass
    def with_structured_output(self, schema): return FauxStructure(schema)
    def invoke(self, messages):
        if isinstance(messages, str):
            return Msg("Paris")
        ids = re.findall(r"\[((?:ART|B\d{2})-\d{2}|DIM)\]", messages[-1][1])
        return Msg(f"Nous proposons une organisation adaptée [{ids[0]}].\n\nSecond paragraphe [{ids[-1]}].")

lo = types.ModuleType("langchain_ollama"); lo.ChatOllama = ChatOllama
sys.modules["langchain_ollama"] = lo

# --- LangChain : décorateur tool et agent simulés -------------------------------
lct = types.ModuleType("langchain_core.tools")
def tool(f):
    f.invoke = lambda arg: f(**arg) if isinstance(arg, dict) else f(arg)
    return f
lct.tool = tool
sys.modules["langchain_core"] = types.ModuleType("langchain_core")
sys.modules["langchain_core.tools"] = lct

la = types.ModuleType("langchain.agents")
class FauxAgent:
    def __init__(self, tools): self.tools = {t.__name__: t for t in tools}
    def invoke(self, entree, config):
        q = entree["messages"][0]["content"]
        if "budget prévisionnel" in q or "Cergy" in q:
            rep = "Je n'ai pas trouvé cette information dans les documents."
        elif "rotations" in q:
            rep = self.tools["dimensionnement"](flux=q)
        else:
            rep = self.tools["chercher_dce"](q)[:300]
        return {"messages": [Msg(q, "human"), Msg(rep)]}
la.create_agent = lambda model, tools, system_prompt, checkpointer: FauxAgent(tools)
sys.modules["langchain"] = types.ModuleType("langchain")
sys.modules["langchain.agents"] = la
lg = types.ModuleType("langgraph.checkpoint.memory"); lg.InMemorySaver = lambda: None
sys.modules["langgraph"] = types.ModuleType("langgraph")
sys.modules["langgraph.checkpoint"] = types.ModuleType("langgraph.checkpoint")
sys.modules["langgraph.checkpoint.memory"] = lg

import matplotlib
matplotlib.use("Agg")
code = (ICI / "toutes_cellules.py").read_text(encoding="utf-8")
code = re.sub(r"subprocess\.Popen\(.*\)\n", "", code)
code = code.replace('requests.get("http://localhost:11434", timeout=5)', "pass")
code = "import time\ndisplay = print\n" + code
exec(compile(code, "notebook", "exec"), {"__name__": "__main__"})
print("\n=== EXÉCUTION COMPLÈTE OK ===")
