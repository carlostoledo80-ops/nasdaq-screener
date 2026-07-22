import sys
import traceback

import pandas as pd

from nasdaq_screener.alert import dispatch
from nasdaq_screener.backtest import walk_forward
from nasdaq_screener.config import CONFIG
from nasdaq_screener.data import make_provider
from nasdaq_screener.universe import get_ndx100


def veredicto(valid, n_rows):
    if n_rows == 0:
        return ("SIN DATOS",
                "No se pudo descargar historial suficiente. Reintenta mas tarde.")
    if valid.empty:
        return ("SIN CONDICIONES FAVORABLES",
                "El motor no hallo ninguna condicion que superara la tasa base "
                "ni siquiera en el tramo de entrenamiento. Traducido: no hay "
                "senal que validar. NO operes con esto.")
    if len(valid) < 5:
        return ("MUESTRA INSUFICIENTE",
                "Muy pocas acciones produjeron dias marcados fuera de muestra. "
                "No alcanza para concluir nada.")

    ml = valid["oos_lift"].mean()
    sp = (valid["oos_lift"] > 0).mean()
    mn = valid["net_expectancy"].mean()

    if ml > 0.02 and sp > 0.55 and mn > 0:
        return ("EVIDENCIA DEBIL A FAVOR",
                "Las condiciones aguantaron fuera de muestra y la expectativa "
                "neta es positiva. OJO: es UN corte sobre ~2 anos. Es una luz "
                "verde tenue, no una confirmacion. Arranca con tamano minimo.")
    if ml <= 0 or mn <= 0:
        return ("SIN EVIDENCIA - NO OPERAR",
                "La ventaja no sobrevivio fuera de muestra o se la comen los "
                "costos. Los porcentajes de la alerta diaria son ruido. "
                "Sirve como informacion de contexto, no como senal.")
    return ("AMBIGUO",
            "Hay algo, pero no lo suficiente para arriesgar capital. "
            "Trata la alerta como contexto, no como senal.")


def main():
    cfg = CONFIG
    prov = make_provider(cfg)
    tickers = get_ndx100()
    print(f"Validando {len(tickers)} acciones...", flush=True)

    rows = []
    for i, tk in enumerate(tickers, 1):
        try:
            r = walk_forward(tk, prov, cfg)
            if r is not None:
                rows.append(r.__dict__)
        except Exception as e:
            print(f"  {tk}: fallo ({e})", flush=True)
        if i % 20 == 0:
            print(f"  ...{i}/{len(tickers)}", flush=True)

    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["oos_lift"]) if not df.empty else pd.DataFrame()
    titulo, explicacion = veredicto(valid, len(rows))

    L = []
    L.append("VALIDACION WALK-FORWARD - Screener Nasdaq 100")
    L.append("=" * 34)
    L.append(f"Acciones con historial usable: {len(rows)}")
    umbral = f"{100 * cfg.up_threshold:.2f}".rstrip("0").rstrip(".")
    L.append(f"Evento validado: 1a hora >= apertura +{umbral}%")
    L.append("")
    if not valid.empty:
        L.append(f"Acciones con dias marcados OOS : {len(valid)}")
        L.append(f"Lift medio fuera de muestra    : {valid['oos_lift'].mean() * 100:+.1f} pp")
        L.append(f"Mediana del lift               : {valid['oos_lift'].median() * 100:+.1f} pp")
        L.append(f"% de acciones con lift > 0     : {(valid['oos_lift'] > 0).mean() * 100:.0f}%")
        L.append(f"Expectativa bruta por operacion: {valid['gross_expectancy'].mean() * 100:+.2f}%")
        L.append(f"Costo asumido ida y vuelta     : {cfg.roundtrip_cost * 100:.2f}%")
        L.append(f"Expectativa NETA por operacion : {valid['net_expectancy'].mean() * 100:+.2f}%")
        L.append("")
    L.append("-" * 34)
    L.append(f"VEREDICTO: {titulo}")
    L.append("")
    L.append(explicacion)
    L.append("-" * 34)
    L.append("Metodo: entrenar con el 60% mas antiguo, medir en el 40% final "
             "que el modelo nunca vio. No es asesoria de inversion.")

    dispatch("\n".join(L), cfg)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        traceback.print_exc()
        try:
            dispatch(f"Validacion walk-forward fallo: {e}", CONFIG)
        except Exception:
            pass
        sys.exit(1)
