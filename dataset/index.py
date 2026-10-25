from enum import Enum

GLOSSARY_FILE_FORMAT = "{project}_glossary.csv"
TEST_DATA_FILE_FORMAT = "{project}_test-data.csv"


class DatasetProject(Enum):
    ELSFORK = "elsfork"
    GAMMA_J = "gamma-j"
    GEMINI = "gemini"
    HOME = "home"
    INVENTORY = "inventory"
    NASA = "nasa"
    PEPPOL = "peppol"
    SCEAPI = "sceapi"