import unittest
from src.analyze_sentiment import analizar_sentimiento

class TestAnalisisSentimiento(unittest.TestCase):

    def test_analizar_sentimiento(self):
        texto = "The dollar index has fallen dramatically this week, and economic outlook is pessimistic."
        resultado = analizar_sentimiento(texto)
        self.assertIn("label", resultado[0])
        self.assertIn("score", resultado[0])
        self.assertIn(resultado[0]['label'].lower(), ['positive', 'negative', 'neutral'])

if __name__ == "__main__":
    unittest.main()
