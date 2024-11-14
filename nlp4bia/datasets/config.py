import os

ORIG_VERSION = "raw"
DS_COLUMNS =  ["filenameid", "mention_class", "span", "code", "sem_rel", "is_abbreviation", "is_composite", "needs_context", "extension_esp"]

try:
    NLP4BIA_DATA_PATH = os.environ["NLP4BIA_DATA_PATH"]
except KeyError:
    NLP4BIA_DATA_PATH = os.path.join(os.path.expanduser("~"), ".nlp4bia")