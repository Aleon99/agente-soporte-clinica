import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import herramientas
from ml import db
from ml.limpieza import limpiar_texto
from ml.priorizacion import priorizar


class PriorizacionTests(unittest.TestCase):
    def test_matriz_de_arriba_hacia_abajo(self):
        self.assertEqual(priorizar("paneles_totems", "administrativo"), ("critica", 1))
        self.assertEqual(priorizar("impresion", "consultorio"), ("alta", 2))
        self.assertEqual(priorizar("aplicativos", "triaje"), ("media", 3))
        self.assertEqual(priorizar("encendido", "counter"), ("media", 3))
        self.assertEqual(priorizar("otros", "administrativo"), ("baja", 4))

    def test_area_fuera_de_catalogo(self):
        with self.assertRaises(ValueError):
            priorizar("impresion", "laboratorio")


class LimpiezaTests(unittest.TestCase):
    def test_reglas(self):
        self.assertEqual(
            limpiar_texto("¡La PC 192.168.1.20 NO prende! avisar a soporte@clinica.pe, anexo 2045 (año)"),
            "la pc <ip> no prende avisar a <correo> anexo <num> año",
        )


class HerramientasTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        ruta = Path(self.tmp.name) / "tickets.db"
        self.parches = [
            patch.object(db, "RUTA_DB", ruta),
        ]
        for p in self.parches:
            p.start()

    def tearDown(self):
        for p in self.parches:
            p.stop()
        self.tmp.cleanup()

    def _clasificacion(self, tipo, prob):
        return {"tipo_predicho": tipo, "probabilidad": prob, "revision_tecnico": int(prob < 0.6),
                "umbral": 0.6, "terminos_clave": ["tótem", "turno"], "version_modelo": "prueba"}

    def test_valida_area_y_sede(self):
        r = herramientas.registrar_ticket.func("no imprime", "laboratorio", "miraflores")
        self.assertIn("error", r)
        r = herramientas.registrar_ticket.func("no imprime", "counter", "callao")
        self.assertIn("error", r)
        self.assertEqual(herramientas.validar_area("Admisión"), "counter")
        self.assertEqual(herramientas.validar_area("Contabilidad"), "administrativo")
        self.assertIsNone(herramientas.validar_area("laboratorio"))
        self.assertEqual(herramientas.validar_sede("San Isidro"), "san_isidro")

    def test_registra_prioriza_y_ordena(self):
        with patch.object(herramientas, "clasificar",
                          side_effect=[self._clasificacion("impresion", 0.9),
                                       self._clasificacion("paneles_totems", 0.4)]):
            primero = herramientas.registrar_ticket.func("no imprime", "administrativo", "surco")
            segundo = herramientas.registrar_ticket.func("tótem caído", "counter", "surco")
        self.assertEqual(primero["prioridad"], "baja")
        self.assertEqual(segundo["prioridad"], "critica")
        # Con baja confianza se marca para revisión, pero igual entra a la cola.
        self.assertEqual(segundo["revision_tecnico"], 1)
        cola = db.listar_cola("surco")
        self.assertEqual([t["id"] for t in cola], [segundo["id"], primero["id"]])
        self.assertIn("| 1 | %d | critica" % segundo["id"], herramientas.ver_cola.func("surco"))

    def test_confirmar_tipo(self):
        with patch.object(herramientas, "clasificar", return_value=self._clasificacion("otros", 0.8)):
            r = herramientas.registrar_ticket.func("no hay red", "triaje", "la_molina")
        self.assertIn("error", herramientas.confirmar_tipo.func(r["id"], "teclado"))
        ok = herramientas.confirmar_tipo.func(r["id"], "aplicativos")
        self.assertFalse(ok["coincide"])
        self.assertEqual(db.obtener_ticket(r["id"])["tipo_real"], "aplicativos")


if __name__ == "__main__":
    unittest.main()
