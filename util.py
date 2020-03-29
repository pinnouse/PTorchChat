"""
This module provides some utility functions to work iwth and parse the corpus
for the bot.
"""
from configparser import ConfigParser
import re


def config() -> ConfigParser:
    """
    Returns the config in a dict<section, dict<k, v>> where k is the key name
    and v is the config value.
    """
    cfg = ConfigParser()
    cfg.read('config.ini')
    return cfg

def normalize_string(string_to_normalize: str) -> str:
    """
    Takes an input string and normalizes it to a uniform understanding for
    Python.
    """
    string_to_normalize = re.sub(r"([.!?:()][\\s]*)", r" \1 ", string_to_normalize)
    string_to_normalize = re.sub(r"\s+", r" ", string_to_normalize).strip()
    return string_to_normalize.lower()
