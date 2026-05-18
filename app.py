from flask import Flask, render_template, request, jsonify
from db import get_connection

app = Flask(__name__)

@app.route("/")
def index():
    conn = get_connection()
    cursor = conn.cursor(as_dict=True)

    cursor.execute("""
        SELECT g.departamento, ROUND(AVG(h.valor_medicion), 2) as promedio
        FROM Hechos_Climaticos h
        JOIN Dim_Geografia g ON h.id_geo = g.id_geo
        JOIN Dim_Parametro p ON h.id_parametro = p.id_parametro
        WHERE p.parametro = 'PRECIPITACIÓN'
        GROUP BY g.departamento ORDER BY promedio DESC
    """)
    precip_depto = cursor.fetchall()

    cursor.execute("""
        SELECT g.departamento, ROUND(AVG(h.valor_medicion), 2) as promedio
        FROM Hechos_Climaticos h
        JOIN Dim_Geografia g ON h.id_geo = g.id_geo
        JOIN Dim_Parametro p ON h.id_parametro = p.id_parametro
        WHERE p.parametro = 'TEMPERATURA MEDIA'
        GROUP BY g.departamento ORDER BY promedio DESC
    """)
    temp_media_depto = cursor.fetchall()

    cursor.execute("""
        SELECT g.departamento, ROUND(AVG(h.valor_medicion), 2) as promedio
        FROM Hechos_Climaticos h
        JOIN Dim_Geografia g ON h.id_geo = g.id_geo
        JOIN Dim_Parametro p ON h.id_parametro = p.id_parametro
        WHERE p.parametro = 'TEMPERATURA MÁXIMA'
        GROUP BY g.departamento ORDER BY promedio DESC
    """)
    temp_max_depto = cursor.fetchall()

    cursor.execute("""
        SELECT g.departamento, ROUND(AVG(h.valor_medicion), 2) as promedio
        FROM Hechos_Climaticos h
        JOIN Dim_Geografia g ON h.id_geo = g.id_geo
        JOIN Dim_Parametro p ON h.id_parametro = p.id_parametro
        WHERE p.parametro = 'TEMPERATURA MÍNIMA'
        GROUP BY g.departamento ORDER BY promedio DESC
    """)
    temp_min_depto = cursor.fetchall()

    cursor.execute("""
        SELECT TOP 10 e.estacion, g.departamento, ROUND(AVG(h.valor_medicion), 2) as promedio
        FROM Hechos_Climaticos h
        JOIN Dim_Estacion e ON h.id_estacion = e.id_estacion
        JOIN Dim_Geografia g ON h.id_geo = g.id_geo
        JOIN Dim_Parametro p ON h.id_parametro = p.id_parametro
        WHERE p.parametro = 'No. DE DIAS CON LLUVIA'
        GROUP BY e.estacion, g.departamento ORDER BY promedio DESC
    """)
    dias_lluvia_top = cursor.fetchall()

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

    return render_template("index.html",
        precip_depto=precip_depto,
        temp_media_depto=temp_media_depto,
        temp_max_depto=temp_max_depto,
        temp_min_depto=temp_min_depto,
        dias_lluvia_top=dias_lluvia_top,
        departamentos=departamentos,
        parametros=parametros,
        periodos=periodos
    )

@app.route("/consulta_filtro", methods=["POST"])
def consulta_filtro():
    data = request.json
    parametro = data.get("parametro", "")
    departamento = data.get("departamento", "")
    periodo = data.get("periodo", "")

    conditions = ["p.parametro = %s"]
    params = [parametro]

    if departamento:
        conditions.append("g.departamento = %s")
        params.append(departamento)
    if periodo:
        conditions.append("per.periodo = %s")
        params.append(periodo)

    query = f"""
        SELECT g.departamento, per.periodo, e.estacion,
               ROUND(AVG(h.valor_medicion), 2) as promedio
        FROM Hechos_Climaticos h
        JOIN Dim_Geografia g ON h.id_geo = g.id_geo
        JOIN Dim_Parametro p ON h.id_parametro = p.id_parametro
        JOIN Dim_Periodo per ON h.id_periodo = per.id_periodo
        JOIN Dim_Estacion e ON h.id_estacion = e.id_estacion
        WHERE {' AND '.join(conditions)}
        GROUP BY g.departamento, per.periodo, e.estacion
        ORDER BY promedio DESC
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
    app.run(debug=True)