import unittest
import pandas as pd

class TestCargaDatosHistoricos(unittest.TestCase):

    def setUp(self):
        self.ruta_csv = "data/market/eurusd_historico_1min.csv"
        self.df = pd.read_csv(self.ruta_csv)

    def test_columnas_esperadas(self):
        columnas_esperadas = ["datetime", "open", "high", "low", "close", "volume"]
        self.assertListEqual(list(self.df.columns), columnas_esperadas)

    def test_sin_valores_nulos(self):
        self.assertFalse(self.df.isnull().any().any(), "Existen valores nulos en el DataFrame")

    def test_tipos_de_datos_correctos(self):
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(pd.to_datetime(self.df["datetime"], errors="coerce")))
        self.assertTrue(pd.api.types.is_float_dtype(self.df["open"]))
        self.assertTrue(pd.api.types.is_float_dtype(self.df["high"]))
        self.assertTrue(pd.api.types.is_float_dtype(self.df["low"]))
        self.assertTrue(pd.api.types.is_float_dtype(self.df["close"]))
        self.assertTrue(pd.api.types.is_float_dtype(self.df["volume"]) or pd.api.types.is_integer_dtype(self.df["volume"]))

    def test_sin_fechas_duplicadas(self):
        duplicados = self.df.duplicated(subset=["datetime"]).sum()
        self.assertEqual(duplicados, 0, f"Hay {duplicados} fechas duplicadas en 'datetime'")

if __name__ == "__main__":
    unittest.main()
