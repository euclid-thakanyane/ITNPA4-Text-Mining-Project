"""Text denoising utilities shared by the notebook and the Streamlit app."""
import re, unicodedata
import spacy

_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
STOPWORDS = _nlp.Defaults.stop_words | {
    "paper", "study", "article", "result", "results", "author", "authors", "use", "using", "used",
    "propose", "proposed", "show", "find", "finding", "also", "however", "et", "al", "elsevier",
    "wiley", "son", "ltd", "inc", "right", "reserve", "copyright", "journal"}

COPYRIGHT = re.compile(r"(copyright|©|\(c\))[^.]*(\.|$)", re.I)
LATEX_MATH = re.compile(r"\$[^$]*\$")
LATEX_CMD = re.compile(r"\\[a-zA-Z]+(\{[^}]*\})?")
HTML = re.compile(r"<[^>]+>|&[a-z]+;")
URL = re.compile(r"https?://\S+|www\.\S+|doi:\s*\S+", re.I)
EMAIL = re.compile(r"\S+@\S+")
NON_ALPHA = re.compile(r"[^a-z\s]")
SPACES = re.compile(r"\s+")

def denoise(text: str) -> str:
    """Strip markup/boilerplate noise, normalise unicode, lowercase and remove numbers & punctuation."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", " ")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    for pat in (COPYRIGHT, LATEX_MATH, LATEX_CMD, HTML, URL, EMAIL):
        text = pat.sub(" ", text)
    text = text.lower().replace("-", " ")
    text = NON_ALPHA.sub(" ", text)
    return SPACES.sub(" ", text).strip()

def normalise(text: str) -> str:
    """Lemmatise with spaCy and drop stop words / very short tokens."""
    doc = _nlp(text)
    return " ".join(t.lemma_ for t in doc if t.lemma_ not in STOPWORDS and len(t.lemma_) > 2 and t.is_alpha)

def clean_text(text: str) -> str:
    return normalise(denoise(text))

def clean_many(texts, batch_size=64):
    den = [denoise(t) for t in texts]
    out = []
    for doc in _nlp.pipe(den, batch_size=batch_size):
        out.append(" ".join(t.lemma_ for t in doc if t.lemma_ not in STOPWORDS and len(t.lemma_) > 2 and t.is_alpha))
    return out
