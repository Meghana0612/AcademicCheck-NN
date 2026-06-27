from langdetect import detect
from deep_translator import GoogleTranslator

def translate_to_english(text):
    try:
        language = detect(text)

        if language == "en":
            return text, "English"

        translated = GoogleTranslator(
            source="auto",
            target="en"
        ).translate(text)

        return translated, language

    except Exception:
        return text, "Unknown"