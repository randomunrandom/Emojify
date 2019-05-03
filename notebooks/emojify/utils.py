import os
import json

def save_text(text: str, filename: str):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)
        
        
def read_text(filename: str) -> str:
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()