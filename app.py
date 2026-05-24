from flask import Flask, render_template, request, jsonify
from db import get_connection
import threading

try:
    from clustering_spark import ejecutar_clustering
except ImportError as err:
    err_msg = str(err)
    def ejecutar_clustering():
        return {"exito": False, "error": f"Import error: {err_msg}"}
    
app = Flask(__name__)

_clustering_cache = None
_clustering_lock = threading.Lock()

def get_clustering_data():
    global _clustering_cache
    if _clustering_cache is not None:
        return _clustering_cache
    with _clustering_lock:
        if _clustering_cache is None:
            try:
                _clustering_cache = ejecutar_clustering()
            except Exception as e:
                _clustering_cache = {"exito": False, "error": str(e)}
    return _clustering_cache

def get_dashboard_context():
    conn = get_connection()
    cursor = conn.cursor(as_dict=True)

    # CONSULTA 1: Variación de precipitación por departamento, mes y período
    cursor.execute("""
        SELECT g.departamento, t.nombre_mes, t.num_mes,
               per.periodo,
               ROUND(AVG(h.valor_medicion), 2) as promedio
        FROM Hechos_Climaticos h
        JOIN Dim_Geografia g ON h.id_geo = g.id_geo
        JOIN Dim_Tiempo t ON h.id_tiempo = t.id_tiempo
        JOIN Dim_Periodo per ON h.id_periodo = per.id_periodo
        JOIN Dim_Parametro p ON h.id_parametro = p.id_parametro
        WHERE p.parametro = 'PRECIPITACIÓN'
        GROUP BY g.departamento, t.nombre_mes, t.num_mes, per.periodo
        ORDER BY g.departamento, ISNULL(t.num_mes, 99), per.periodo
    """)
    precip_por_mes_periodo = cursor.fetchall()

    # CONSULTA 2: Amplitud térmica por estación y altitud
    cursor.execute("""
        SELECT TOP 15 e.estacion, e.altitud, g.departamento,
               ROUND(MAX(CASE WHEN p.parametro='TEMPERATURA MÁXIMA' THEN h.valor_medicion END), 2) as temp_max,
               ROUND(MIN(CASE WHEN p.parametro='TEMPERATURA MÍNIMA' THEN h.valor_medicion END), 2) as temp_min,
               ROUND(MAX(CASE WHEN p.parametro='TEMPERATURA MÁXIMA' THEN h.valor_medicion END) -
                     MIN(CASE WHEN p.parametro='TEMPERATURA MÍNIMA' THEN h.valor_medicion END), 2) as amplitud_termica
        FROM Hechos_Climaticos h
        JOIN Dim_Estacion e ON h.id_estacion = e.id_estacion
        JOIN Dim_Geografia g ON h.id_geo = g.id_geo
        JOIN Dim_Parametro p ON h.id_parametro = p.id_parametro
        WHERE p.parametro IN ('TEMPERATURA MÁXIMA','TEMPERATURA MÍNIMA')
        GROUP BY e.estacion, e.altitud, g.departamento
        HAVING MAX(CASE WHEN p.parametro='TEMPERATURA MÁXIMA' THEN h.valor_medicion END) IS NOT NULL
        ORDER BY amplitud_termica DESC
    """)
    amplitud_termica = cursor.fetchall()

    # CONSULTA 3: Índice combinado por departamento (4 variables)
    cursor.execute("""
        SELECT g.departamento,
               ROUND(AVG(CASE WHEN p.parametro='PRECIPITACIÓN' THEN h.valor_medicion END), 2) as precip_promedio,
               ROUND(AVG(CASE WHEN p.parametro='TEMPERATURA MEDIA' THEN h.valor_medicion END), 2) as temp_promedio,
               ROUND(AVG(CASE WHEN p.parametro='No. DE DIAS CON LLUVIA' THEN h.valor_medicion END), 2) as dias_lluvia,
               ROUND(AVG(CAST(e.altitud AS FLOAT)), 0) as altitud_media
        FROM Hechos_Climaticos h
        JOIN Dim_Geografia g ON h.id_geo = g.id_geo
        JOIN Dim_Estacion e ON h.id_estacion = e.id_estacion
        JOIN Dim_Parametro p ON h.id_parametro = p.id_parametro
        WHERE p.parametro IN ('PRECIPITACIÓN','TEMPERATURA MEDIA','No. DE DIAS CON LLUVIA')
        GROUP BY g.departamento
        ORDER BY precip_promedio DESC
    """)
    indice_combinado = cursor.fetchall()

    # CONSULTA 4: Tendencia climática mes a mes entre períodos
    cursor.execute("""
        SELECT per.periodo, t.nombre_mes, t.num_mes, p.parametro,
               ROUND(AVG(h.valor_medicion), 2) as valor_promedio
        FROM Hechos_Climaticos h
        JOIN Dim_Periodo per ON h.id_periodo = per.id_periodo
        JOIN Dim_Tiempo t ON h.id_tiempo = t.id_tiempo
        JOIN Dim_Parametro p ON h.id_parametro = p.id_parametro
        WHERE p.parametro IN ('PRECIPITACIÓN','TEMPERATURA MEDIA')
        GROUP BY per.periodo, t.nombre_mes, t.num_mes, p.parametro
        ORDER BY p.parametro, t.num_mes, per.periodo
    """)
    tendencia_periodos = cursor.fetchall()

    # CONSULTA 5: Departamentos más lluviosos por período y su mes pico
    cursor.execute("""
       
    WITH PrecipPorDeptoMes AS (
        SELECT g.departamento, t.nombre_mes, t.num_mes, per.periodo,
               ROUND(AVG(h.valor_medicion), 2) as precipitacion_promedio,
               COUNT(DISTINCT e.id_estacion) as num_estaciones
        FROM Hechos_Climaticos h
        JOIN Dim_Tiempo t ON h.id_tiempo = t.id_tiempo
        JOIN Dim_Periodo per ON h.id_periodo = per.id_periodo
        JOIN Dim_Geografia g ON h.id_geo = g.id_geo
        JOIN Dim_Estacion e ON h.id_estacion = e.id_estacion
        JOIN Dim_Parametro p ON h.id_parametro = p.id_parametro
        WHERE p.parametro = 'PRECIPITACIÓN'
        AND t.nombre_mes != 'Anual'
        GROUP BY g.departamento, t.nombre_mes, t.num_mes, per.periodo
    ),
    MesPicoPorDepto AS (
        SELECT departamento, nombre_mes, periodo,
               precipitacion_promedio, num_estaciones,
               ROW_NUMBER() OVER (
                   PARTITION BY periodo, departamento
                   ORDER BY precipitacion_promedio DESC
               ) as rk_mes
        FROM PrecipPorDeptoMes
    ),
    Top3PorPeriodo AS (
        SELECT departamento, nombre_mes, periodo,
               precipitacion_promedio, num_estaciones,
               ROW_NUMBER() OVER (
                   PARTITION BY periodo
                   ORDER BY precipitacion_promedio DESC
               ) as rk_depto
        FROM MesPicoPorDepto
        WHERE rk_mes = 1
    )
    SELECT departamento, nombre_mes, periodo,
           precipitacion_promedio, num_estaciones
    FROM Top3PorPeriodo
    WHERE rk_depto <= 3
    ORDER BY periodo, rk_depto
    """)
    meses_lluviosos = cursor.fetchall()

    cursor.execute("SELECT DISTINCT departamento FROM Dim_Geografia ORDER BY departamento")
    departamentos = [r['departamento'] for r in cursor.fetchall()]

    cursor.execute("""
        SELECT DISTINCT parametro FROM Dim_Parametro 
        WHERE parametro != 'BRILLO SOLAR' ORDER BY parametro
    """)
    parametros = [r['parametro'] for r in cursor.fetchall()]

    cursor.execute("SELECT DISTINCT periodo FROM Dim_Periodo ORDER BY periodo")
    periodos = [r['periodo'] for r in cursor.fetchall()]

    conn.close()

    return dict(
        precip_por_mes_periodo=precip_por_mes_periodo,
        amplitud_termica=amplitud_termica,
        indice_combinado=indice_combinado,
        tendencia_periodos=tendencia_periodos,
        meses_lluviosos=meses_lluviosos,
        departamentos=departamentos,
        parametros=parametros,
        periodos=periodos
    )

@app.route("/")
def index():
    try:
        ctx = get_dashboard_context()
    except Exception as e:
        return f"<pre>ERROR en get_dashboard_context:\n{e}</pre>", 500
    ctx["clustering_data"] = {"exito": False, "estatico": True}
    return render_template("index.html", **ctx)

@app.route("/consulta_filtro", methods=["POST"])
def consulta_filtro():
    data = request.json
    parametro = data.get("parametro", "")
    departamento = data.get("departamento", "")
    periodo = data.get("periodo", "")

    conditions = ["p.parametro = %$"]
    params = [parametro]

    if departamento:
        conditions.append("g.departamento = %$")
        params.append(departamento)
    if periodo:
        conditions.append("per.periodo = %$")
        params.append(periodo)

    # Siempre cruza: Geografia x Tiempo x Periodo x Parametro
    query = f"""
        SELECT g.departamento, t.nombre_mes, t.num_mes,
               per.periodo, p.parametro, p.unidad_medida,
               ROUND(AVG(h.valor_medicion), 2) as promedio,
               ROUND(MAX(h.valor_medicion), 2) as maximo,
               ROUND(MIN(h.valor_medicion), 2) as minimo,
               COUNT(DISTINCT e.id_estacion) as num_estaciones
        FROM Hechos_Climaticos h
        JOIN Dim_Geografia g ON h.id_geo = g.id_geo
        JOIN Dim_Parametro p ON h.id_parametro = p.id_parametro
        JOIN Dim_Periodo per ON h.id_periodo = per.id_periodo
        JOIN Dim_Estacion e ON h.id_estacion = e.id_estacion
        JOIN Dim_Tiempo t ON h.id_tiempo = t.id_tiempo
        WHERE {' AND '.join(conditions)}
        GROUP BY g.departamento, t.nombre_mes, t.num_mes,
                 per.periodo, p.parametro, p.unidad_medida
        ORDER BY g.departamento, t.num_mes, per.periodo
    """
    conn = get_connection()
    cursor = conn.cursor(as_dict=True)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return jsonify(rows)

@app.route("/consulta_libre", methods=["POST"])
def consulta_libre():
    data = request.json
    sql = data.get("sql", "").strip()
    if not sql.upper().startswith("SELECT"):
        return jsonify({"error": "Solo se permiten consultas SELECT"}), 400
    try:
        conn = get_connection()
        cursor = conn.cursor(as_dict=True)
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)