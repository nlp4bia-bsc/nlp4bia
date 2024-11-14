import os

ORIG_VERSION = "raw"
DS_COLUMNS =  ["filenameid", "mention_class", "span", "code", "sem_rel", "is_abbreviation", "is_composite", "needs_context", "extension_esp"]
GZ_COLUMNS = ["code", "language", "term", "semantic_tag", "mainterm"]

try:
    NLP4BIA_DATA_PATH = os.environ["NLP4BIA_DATA_PATH"]
except KeyError:
    NLP4BIA_DATA_PATH = os.path.join(os.path.expanduser("~"), ".nlp4bia")