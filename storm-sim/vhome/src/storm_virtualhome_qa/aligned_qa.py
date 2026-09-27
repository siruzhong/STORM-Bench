"""Normalize simulator object names for question text."""
import re

TERMS = {'lightswitch': 'light switch', 'tablelamp': 'table lamp',
         'facecream': 'face cream', 'barsoap': 'bar of soap',
         'poundcake': 'pound cake', 'coffeetable': 'coffee table',
         'mousemat': 'mouse pad', 'kitchencabinet': 'kitchen cabinet'}


def name(text):
    for old, new in TERMS.items():
        text = re.sub(r'\b' + old + r'\b', new, text)
    return 'toilet lid' if text == 'toilet' else text
