# kg_canon.py
from rapidfuzz import process as rfz, fuzz

class Canon:
    def __init__(self, threshold=90):
        self.names = []
        self.alias = {}
        self.thresh = threshold

    def canonical(self, name: str) -> str:
        if name in self.alias: return self.alias[name]
        if not self.names:
            self.names.append(name); self.alias[name]=name; return name
        match, score, _ = rfz.extractOne(name, self.names, scorer=fuzz.WRatio)
        if score >= self.thresh:
            self.alias[name] = match
            return match
        self.names.append(name); self.alias[name]=name
        return name

def coarse_type(name: str) -> str:
    n = (name or "").lower()
    if any(k in n for k in ["hospital","university","institute","ltd","pty","inc","foundation","org"]): return "Org"
    if any(k in n for k in ["vitamin","protein","calcium","iron","fiber","carb","fat","sodium"]): return "Nutrient"
    if any(k in n for k in ["diabetes","cancer","rickets","anemia","flu","hypertension","asthma"]): return "Condition"
    if any(k in n for k in ["australia","sydney","usa","india","uk"]): return "Place"
    return "Concept"
