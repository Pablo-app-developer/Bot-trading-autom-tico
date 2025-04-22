import unittest
from src.preprocess_news import limpiar_texto

class TestPreprocesamientoNoticias(unittest.TestCase):

    def test_limpiar_html(self):
        texto = "<p>Hola <b>mundo</b></p>"
        resultado = limpiar_texto(texto)
        self.assertEqual(resultado, "hola mundo")

    def test_limpiar_url(self):
        texto = "visita https://example.com para más info"
        resultado = limpiar_texto(texto)
        self.assertEqual(resultado, "visita para mas info")  # <-- sin tilde en "más"


    def test_caracteres_especiales(self):
        texto = "¿Cómo estás? ¡Excelente! @#"
        resultado = limpiar_texto(texto)
        self.assertEqual(resultado, "como estas excelente")

    def test_texto_vacio(self):
        texto = None
        resultado = limpiar_texto(texto)
        self.assertEqual(resultado, "")

if __name__ == "__main__":
    unittest.main()
