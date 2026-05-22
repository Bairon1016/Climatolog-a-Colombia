"""
ANÁLISIS DE CONOCIMIENTO PROFUNDO — FASE 4
Climatología Colombia · PySpark + scikit-learn + scipy

Pipeline completo:
  0.  Carga Excel (pandas) → CSV temporal
  1.  Benchmark 3 configs Spark
  2.  SparkSession principal
  3.  Carga CSV en Spark DataFrame
  4.  VectorAssembler + StandardScaler (Spark)
  5.  K-Means MLlib · Elbow + Silhouette
  6.  PCA (Spark MLlib + sklearn para plots)
  7.  DBSCAN (sklearn, auto-eps vía k-NN)
  8.  Clustering Jerárquico (scipy Ward + dendrograma)
  9.  Detección de anomalías: Isolation Forest + LOF
 10.  Modelo predictivo: Random Forest + cross-validation
 11.  Visualizaciones matplotlib (7 gráficos)
  +   Consultas analíticas Spark SQL
  +   Exportación CSV de resultados
"""

import os
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

# Backend no-GUI antes de cualquier import de pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ---------- PySpark ----------
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# ---------- scikit-learn ----------
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler as SKScaler
from sklearn.decomposition import PCA as SKPCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    f1_score,
)

# ---------- scipy ----------
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# =============================================================================
# CONSTANTES
# =============================================================================

ARCHIVO_DATOS = "static/data/Normales_Climatológicas_de_Colombia_20260425.xlsx"
MESES = ["ENE", "FEB", "MAR", "ABR", "MAY", "JUN", "JUL", "AGO", "SEP", "OCT", "NOV", "DIC"]
COLUMNAS_BASE = ["ALTITUD_m", "LATITUD", "LONGITUD", "ANUAL"]
IMG_DIR = "static/img"

_SPARK_BASE_CONFIG = {
    "spark.driver.memory": "2g",
    "spark.sql.shuffle.partitions": "4",
    "spark.ui.showConsoleProgress": "false",
}

COLORS = [
    "#38bdf8", "#2dd4bf", "#f59e0b", "#a78bfa",
    "#fb7185", "#34d399", "#f97316", "#60a5fa",
]


def _build_spark(master, app_name):
    builder = SparkSession.builder.appName(app_name).master(master)
    for k, v in _SPARK_BASE_CONFIG.items():
        builder = builder.config(k, v)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# =============================================================================
# CLASE PRINCIPAL
# =============================================================================

class AnalisisProfundoSpark:
    """Pipeline completo de conocimiento profundo: Spark + sklearn + scipy."""

    def __init__(self):
        self.spark = None
        self.df_spark = None
        self.df_features = None          # Spark DF con features escaladas
        self.df_pandas = None            # pandas para algoritmos sklearn
        self.df_scaled_np = None         # numpy escalado para sklearn
        self.feature_cols = []           # nombres de columnas de features
        self.modelos = {}
        self.metricas = {}
        self.extras = {}                 # resultados de análisis avanzados
        self.tiempo_inicio = time.time()
        self._tmp_csv_path = None
        self._linkage_matrix = None      # para dendrograma

    # =========================================================================
    # 0. CARGA CON PANDAS (Excel → CSV temporal)
    # =========================================================================

    def cargar_excel_pandas(self):
        print("[0/11] Cargando Excel con pandas...")
        try:
            df_pd = pd.read_excel(ARCHIVO_DATOS)

            # Limpiar nombres de columnas
            df_pd.columns = (
                df_pd.columns.str.strip()
                .str.replace(" ", "_")
                .str.replace("(", "")
                .str.replace(")", "")
            )

            # Filtrar por PRECIPITACIÓN para análisis consistente
            if "PARAMETRO" in df_pd.columns:
                df_precip = df_pd[
                    df_pd["PARAMETRO"].str.contains("PRECIP", case=False, na=False)
                ].copy()
                if len(df_precip) >= 100:
                    df_pd = df_precip
                    print(f"      Filtrado por PRECIPITACIÓN: {len(df_pd):,} registros")

            # Convertir columnas numéricas
            for col in MESES + COLUMNAS_BASE:
                if col in df_pd.columns:
                    df_pd[col] = pd.to_numeric(df_pd[col], errors="coerce")

            # Determinar features disponibles (base + meses)
            self.feature_cols = [
                c for c in COLUMNAS_BASE + MESES if c in df_pd.columns
            ]

            # Filtrar filas con todas las features válidas
            core = [c for c in ["ALTITUD_m", "LATITUD", "LONGITUD", "ANUAL"] if c in df_pd.columns]
            df_pd = df_pd.dropna(subset=core)
            df_full = df_pd.dropna(subset=self.feature_cols)
            if len(df_full) >= 50:
                df_pd = df_full

            self.df_pandas = df_pd.reset_index(drop=True)

            # CSV temporal para Spark
            fd, tmp_path = tempfile.mkstemp(suffix=".csv")
            os.close(fd)
            self._tmp_csv_path = tmp_path
            self.df_pandas.to_csv(self._tmp_csv_path, index=False, encoding="utf-8")

            print(
                f"      {len(self.df_pandas):,} registros, "
                f"{len(self.feature_cols)} features: {self.feature_cols}"
            )
            return True
        except Exception as e:
            print(f"      Error: {e}")
            import traceback; traceback.print_exc()
            return False

    # =========================================================================
    # 1. BENCHMARK DE CONFIGURACIONES
    # =========================================================================

    def _warmup_jvm(self):
        print("      Calentando JVM (pre-benchmark)...")
        try:
            spark_w = _build_spark("local[1]", "Warmup")
            (
                spark_w.read
                .option("header", "true").option("inferSchema", "true")
                .option("encoding", "UTF-8").csv(self._tmp_csv_path)
                .count()
            )
            spark_w.stop()
        except Exception:
            pass

    def _ejecutar_benchmark(self, master_config, descripcion):
        app_safe = master_config.replace("[", "").replace("]", "").replace("*", "N")
        t0 = time.time()
        try:
            spark_b = _build_spark(master_config, f"Bench_{app_safe}")
            df = (
                spark_b.read
                .option("header", "true").option("inferSchema", "true")
                .option("encoding", "UTF-8").csv(self._tmp_csv_path)
                .cache()
            )
            df.count()

            cols_upper = {c.upper(): c for c in df.columns}
            col_dep = next((cols_upper[c] for c in cols_upper if "DEPART" in c), None)

            if col_dep:
                df.groupBy(col_dep).agg(
                    F.count("*"), F.avg("ANUAL"), F.stddev("ANUAL"), F.max("ANUAL")
                ).collect()

            df.withColumn(
                "piso",
                F.when(F.col("ALTITUD_m") < 500,  "Bajo")
                .when(F.col("ALTITUD_m") < 1500,  "Calido")
                .when(F.col("ALTITUD_m") < 2500,  "Templado")
                .when(F.col("ALTITUD_m") < 3500,  "Frio")
                .otherwise("Paramo"),
            ).groupBy("piso").agg(
                F.count("*"), F.avg("ANUAL"), F.avg("ALTITUD_m")
            ).collect()

            df.select(
                F.count("ANUAL"), F.avg("ANUAL"),
                F.stddev("ANUAL"), F.min("ANUAL"), F.max("ANUAL"),
            ).collect()

            spark_b.stop()
        except Exception as e:
            print(f"      Error benchmark {master_config}: {e}")

        tiempo = round(time.time() - t0, 2)
        print(f"      {master_config}: {tiempo}s")
        return {"master": master_config, "descripcion": descripcion, "tiempo_s": tiempo}

    def benchmark_configuraciones(self):
        print("[1/11] Benchmark de configuraciones Spark...")
        self._warmup_jvm()
        configs = [
            ("local[1]", "Solo nodo master (1 hilo)"),
            ("local[2]", "Master + 1 worker"),
            ("local[*]", "Master + todos los workers"),
        ]
        self.metricas["benchmark"] = [self._ejecutar_benchmark(m, d) for m, d in configs]
        print("      Benchmark completo")
        return True

    # =========================================================================
    # 2. SPARKSESSION PRINCIPAL
    # =========================================================================

    def iniciar_spark(self):
        print("[2/11] Iniciando SparkSession principal...")
        try:
            self.spark = _build_spark("local[*]", "AnalisisProfundoColombia")
            print(f"      Spark {self.spark.version} listo")
            return True
        except Exception as e:
            print(f"      Error: {e}")
            return False

    # =========================================================================
    # 3. CARGA EN SPARK
    # =========================================================================

    def cargar_datos(self):
        print("[3/11] Cargando datos en Spark...")
        try:
            self.df_spark = (
                self.spark.read
                .option("header", "true").option("inferSchema", "true")
                .option("encoding", "UTF-8").csv(self._tmp_csv_path)
                .cache()
            )
            n = self.df_spark.count()
            self.metricas["num_registros"] = int(n)
            print(f"      {n:,} registros x {len(self.df_spark.columns)} columnas")
            return True
        except Exception as e:
            print(f"      Error: {e}")
            return False

    # =========================================================================
    # 4. PREPROCESAMIENTO: VectorAssembler + StandardScaler
    # =========================================================================

    def procesar_datos(self):
        print("[4/11] Procesando features (VectorAssembler + StandardScaler)...")
        try:
            spark_cols = set(self.df_spark.columns)
            feat_cols = [c for c in self.feature_cols if c in spark_cols]

            assembler = VectorAssembler(
                inputCols=feat_cols, outputCol="features_raw", handleInvalid="skip"
            )
            df_asm = assembler.transform(self.df_spark)

            scaler = StandardScaler(
                inputCol="features_raw", outputCol="features",
                withMean=True, withStd=True,
            )
            scaler_model = scaler.fit(df_asm)
            self.df_features = (
                scaler_model.transform(df_asm)
                .select("features", *feat_cols)
                .cache()
            )
            self.df_features.count()

            # Versión sklearn (numpy escalado)
            sk_scaler = SKScaler()
            X_raw = self.df_pandas[feat_cols].values
            self.df_scaled_np = sk_scaler.fit_transform(X_raw)

            self.metricas["features_usadas"] = feat_cols
            print(f"      Features activas: {feat_cols}")
            return True
        except Exception as e:
            print(f"      Error: {e}")
            import traceback; traceback.print_exc()
            return False

    # =========================================================================
    # 5. K-MEANS CON SPARK MLlib (Elbow + Silhouette)
    # =========================================================================

    def clustering_kmeans(self, min_k=2, max_k=8):
        print(f"[5/11] K-Means Spark MLlib (k={min_k}..{max_k})...")
        try:
            evaluator = ClusteringEvaluator(
                featuresCol="features", predictionCol="prediction",
                metricName="silhouette", distanceMeasure="squaredEuclidean",
            )
            metricas_k = []
            for k in range(min_k, max_k + 1):
                km = KMeans(
                    featuresCol="features", predictionCol="prediction",
                    k=k, seed=42, initMode="k-means||", maxIter=100,
                )
                modelo = km.fit(self.df_features)
                preds = modelo.transform(self.df_features)
                sil = float(evaluator.evaluate(preds))
                inertia = float(modelo.summary.trainingCost)
                self.modelos[k] = {"modelo": modelo, "predicciones": preds}
                metricas_k.append({
                    "k": k,
                    "silhouette": round(sil, 4),
                    "inertia": round(inertia, 2),
                })
                print(f"      k={k}  Silhouette={sil:.4f}  Inertia={inertia:.2f}")

            self.metricas["por_k"] = metricas_k
            mejor = max(metricas_k, key=lambda m: m["silhouette"])
            self.metricas["mejor_k"] = int(mejor["k"])
            self.metricas["mejor_silhouette"] = float(mejor["silhouette"])
            self.metricas["mejor_inertia"] = float(mejor["inertia"])
            self.metricas["tabla_metricas"] = metricas_k

            # Etiquetas K-Means en pandas
            mejor_k = self.metricas["mejor_k"]
            preds_best = self.modelos[mejor_k]["predicciones"]
            labels_km = [
                int(r["prediction"])
                for r in preds_best.select("prediction").collect()
            ]
            if len(labels_km) == len(self.df_pandas):
                self.df_pandas["cluster_kmeans"] = labels_km
            else:
                from sklearn.cluster import KMeans as SKKMeans
                skm = SKKMeans(n_clusters=mejor_k, random_state=42, n_init=10)
                self.df_pandas["cluster_kmeans"] = skm.fit_predict(self.df_scaled_np)

            # Estadísticas por cluster
            spark_cols = preds_best.columns
            agg_exprs = [F.count("*").alias("registros")]
            if "ANUAL" in spark_cols:
                agg_exprs += [
                    F.round(F.avg("ANUAL"), 2).alias("promedio_mm"),
                    F.round(F.min("ANUAL"), 2).alias("min_mm"),
                    F.round(F.max("ANUAL"), 2).alias("max_mm"),
                ]
            if "ALTITUD_m" in spark_cols:
                agg_exprs.append(F.round(F.avg("ALTITUD_m"), 2).alias("altitud_prom"))

            stats_rows = (
                preds_best.groupBy("prediction")
                .agg(*agg_exprs)
                .orderBy("prediction")
                .collect()
            )

            cluster_stats = []
            cluster_info = {}
            for row in stats_rows:
                cid = int(row["prediction"])
                rd = row.asDict()
                cluster_info[f"Cluster {cid}"] = int(rd["registros"])
                cluster_stats.append({
                    "cluster": cid,
                    "registros": int(rd["registros"]),
                    "promedio_mm": float(rd.get("promedio_mm", 0) or 0),
                    "min_mm": float(rd.get("min_mm", 0) or 0),
                    "max_mm": float(rd.get("max_mm", 0) or 0),
                    "altitud_prom": float(rd.get("altitud_prom", 0) or 0),
                })
            self.metricas["cluster_info"] = cluster_info
            self.metricas["cluster_stats"] = cluster_stats
            return True
        except Exception as e:
            print(f"      Error K-Means: {e}")
            import traceback; traceback.print_exc()
            return False

    # =========================================================================
    # 6. PCA (Spark MLlib + sklearn para proyección)
    # =========================================================================

    def reduccion_pca(self):
        print("[6/11] PCA (Spark MLlib + sklearn)...")
        try:
            feat_cols = self.metricas.get("features_usadas", COLUMNAS_BASE)
            n_comp = min(len(feat_cols), 8)

            # Spark PCA → varianza explicada
            spark_pca = SparkPCA(inputCol="features", outputCol="pca_features", k=n_comp)
            pca_model = spark_pca.fit(self.df_features)
            var_exp = [float(v) for v in pca_model.explainedVariance]
            var_acum = [sum(var_exp[: i + 1]) for i in range(len(var_exp))]
            n_80pct = next(
                (i + 1 for i, v in enumerate(var_acum) if v >= 0.80), n_comp
            )
            print(f"      Var. acumulada: {[round(v,3) for v in var_acum]}")
            print(f"      Componentes para ≥80%: {n_80pct}")

            # sklearn PCA → coordenadas para scatter
            n_sk = min(3, len(feat_cols))
            sk_pca = SKPCA(n_components=n_sk, random_state=42)
            pca_coords = sk_pca.fit_transform(self.df_scaled_np)
            self.df_pandas["PC1"] = pca_coords[:, 0]
            self.df_pandas["PC2"] = pca_coords[:, 1]
            if n_sk > 2:
                self.df_pandas["PC3"] = pca_coords[:, 2]

            # Loadings (variables más importantes por componente)
            loadings = []
            for i, comp in enumerate(sk_pca.components_):
                sorted_idx = np.argsort(np.abs(comp))[::-1]
                loadings.append({
                    "PC": f"PC{i+1}",
                    "varianza_pct": round(float(sk_pca.explained_variance_ratio_[i]) * 100, 1),
                    "top_features": [
                        {"feature": feat_cols[j], "loading": round(float(comp[j]), 3)}
                        for j in sorted_idx[:5]
                        if j < len(feat_cols)
                    ],
                })

            self.extras["pca"] = {
                "varianza_explicada": [round(v, 4) for v in var_exp],
                "varianza_acumulada": [round(v, 4) for v in var_acum],
                "componentes_80pct": n_80pct,
                "loadings": loadings,
            }
            return True
        except Exception as e:
            print(f"      Error PCA: {e}")
            import traceback; traceback.print_exc()
            return False

    # =========================================================================
    # 7. DBSCAN (sklearn, eps auto-tuned por k-NN)
    # =========================================================================

    def clustering_dbscan(self):
        print("[7/11] DBSCAN (sklearn, eps auto-tuned)...")
        try:
            X = self.df_scaled_np
            nbrs = NearestNeighbors(n_neighbors=5).fit(X)
            distances, _ = nbrs.kneighbors(X)
            k_dist = np.sort(distances[:, -1])
            eps = float(np.percentile(k_dist, 90))
            min_samples = max(3, int(np.log(len(X))))

            db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = db.fit_predict(X)
            self.df_pandas["cluster_dbscan"] = labels

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int(np.sum(labels == -1))

            sil_dbscan = 0.0
            mask = labels != -1
            if n_clusters >= 2 and mask.sum() > n_clusters:
                try:
                    sil_dbscan = float(silhouette_score(X[mask], labels[mask]))
                except Exception:
                    pass

            cluster_counts = {
                ("Ruido (-1)" if lbl == -1 else f"Cluster {lbl}"): int(np.sum(labels == lbl))
                for lbl in sorted(set(labels))
            }
            print(
                f"      eps={eps:.3f} minPts={min_samples} → "
                f"{n_clusters} clusters, {n_noise} ruido, Silhouette={sil_dbscan:.4f}"
            )
            self.extras["dbscan"] = {
                "eps": round(eps, 4),
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "pct_noise": round(100 * n_noise / len(labels), 1),
                "silhouette": round(sil_dbscan, 4),
                "cluster_counts": cluster_counts,
            }
            return True
        except Exception as e:
            print(f"      Error DBSCAN: {e}")
            return False

    # =========================================================================
    # 8. CLUSTERING JERÁRQUICO (scipy Ward + dendrograma)
    # =========================================================================

    def clustering_jerarquico(self):
        print("[8/11] Clustering Jerárquico (scipy Ward)...")
        try:
            X = self.df_scaled_np
            best_k = self.metricas.get("mejor_k", 4)

            # Muestra para dendrograma (scipy no escala bien a miles de filas)
            n_sample = min(len(X), 300)
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), n_sample, replace=False)
            Z = linkage(X[idx], method="ward")
            self._linkage_matrix = Z
            self._dendro_n = n_sample

            # Clustering completo con AgglomerativeClustering
            agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
            hier_labels = agg.fit_predict(X)
            self.df_pandas["cluster_jerarquico"] = hier_labels

            sil_hier = 0.0
            if best_k >= 2:
                try:
                    sil_hier = float(silhouette_score(X, hier_labels))
                except Exception:
                    pass

            cluster_counts = {
                f"Cluster {i}": int(np.sum(hier_labels == i)) for i in range(best_k)
            }
            print(f"      Ward k={best_k}: Silhouette={sil_hier:.4f}")
            self.extras["jerarquico"] = {
                "n_clusters": best_k,
                "linkage": "ward",
                "silhouette": round(sil_hier, 4),
                "cluster_counts": cluster_counts,
                "n_muestra": n_sample,
            }
            return True
        except Exception as e:
            print(f"      Error jerárquico: {e}")
            return False

    # =========================================================================
    # 9. DETECCIÓN DE ANOMALÍAS: Isolation Forest + LOF
    # =========================================================================

    def deteccion_anomalias(self):
        print("[9/11] Detección de anomalías (IF + LOF)...")
        try:
            X = self.df_scaled_np
            contamination = 0.05

            # Isolation Forest
            iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
            iso_labels = iso.fit_predict(X)
            iso_scores = iso.score_samples(X)
            self.df_pandas["is_anomaly_if"] = iso_labels == -1
            self.df_pandas["anomaly_score_if"] = iso_scores
            n_iso = int(np.sum(iso_labels == -1))

            # Local Outlier Factor
            lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
            lof_labels = lof.fit_predict(X)
            lof_scores = lof.negative_outlier_factor_
            self.df_pandas["is_anomaly_lof"] = lof_labels == -1
            self.df_pandas["anomaly_score_lof"] = lof_scores
            n_lof = int(np.sum(lof_labels == -1))

            n_both = int(
                (self.df_pandas["is_anomaly_if"] & self.df_pandas["is_anomaly_lof"]).sum()
            )
            print(f"      IF: {n_iso} | LOF: {n_lof} | Ambos: {n_both}")
            self.extras["anomalias"] = {
                "isolation_forest": {
                    "n_anomalias": n_iso,
                    "pct": round(100 * n_iso / len(X), 1),
                    "contamination": contamination,
                },
                "lof": {
                    "n_anomalias": n_lof,
                    "pct": round(100 * n_lof / len(X), 1),
                    "n_neighbors": 20,
                },
                "detectadas_por_ambos": n_both,
                "pct_ambos": round(100 * n_both / len(X), 1),
            }
            return True
        except Exception as e:
            print(f"      Error anomalías: {e}")
            return False

    # =========================================================================
    # 10. MODELO PREDICTIVO: Random Forest sobre clusters K-Means
    # =========================================================================

    def modelo_predictivo(self):
        print("[10/11] Random Forest predictivo...")
        try:
            if "cluster_kmeans" not in self.df_pandas.columns:
                print("      Skip: sin etiquetas K-Means")
                return False

            feat_cols = self.metricas.get("features_usadas", COLUMNAS_BASE)
            X = self.df_scaled_np
            y = self.df_pandas["cluster_kmeans"].values

            # Excluir anomalías del entrenamiento
            if "is_anomaly_if" in self.df_pandas.columns:
                mask = ~self.df_pandas["is_anomaly_if"].values
                X_tr, y_tr = (X[mask], y[mask]) if mask.sum() > 50 else (X, y)
            else:
                X_tr, y_tr = X, y

            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            n_splits = min(5, len(set(y_tr)))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(rf, X_tr, y_tr, cv=cv, scoring="f1_macro")

            rf.fit(X_tr, y_tr)
            y_pred = rf.predict(X)
            acc = float(accuracy_score(y, y_pred))
            f1 = float(f1_score(y, y_pred, average="macro"))

            feat_importance = sorted(
                [
                    {"feature": feat_cols[i], "importance": round(float(rf.feature_importances_[i]), 4)}
                    for i in range(len(feat_cols))
                    if i < len(rf.feature_importances_)
                ],
                key=lambda x: x["importance"],
                reverse=True,
            )
            print(
                f"      Accuracy={acc:.4f}  F1={f1:.4f}  "
                f"CV-F1={cv_scores.mean():.4f}±{cv_scores.std():.4f}"
            )
            self.extras["random_forest"] = {
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4),
                "cv_scores": [round(float(s), 4) for s in cv_scores],
                "cv_mean": round(float(cv_scores.mean()), 4),
                "cv_std": round(float(cv_scores.std()), 4),
                "feature_importance": feat_importance,
                "n_estimators": 100,
                "top_feature": feat_importance[0]["feature"] if feat_importance else "",
            }
            return True
        except Exception as e:
            print(f"      Error RF: {e}")
            import traceback; traceback.print_exc()
            return False

    # =========================================================================
    # CONSULTAS ANALÍTICAS CON SPARK
    # =========================================================================

    def consultas_analiticas(self):
        print("      Consultas analíticas Spark SQL...")
        try:
            analiticas = {}

            stats = self.df_spark.select(
                F.count("ANUAL").alias("total"),
                F.round(F.avg("ANUAL"), 2).alias("prom_anual"),
                F.round(F.stddev("ANUAL"), 2).alias("desv_anual"),
                F.round(F.min("ANUAL"), 2).alias("min_anual"),
                F.round(F.max("ANUAL"), 2).alias("max_anual"),
                F.round(F.avg("ALTITUD_m"), 2).alias("altitud_prom"),
            ).collect()[0]
            analiticas["indicadores_globales"] = {
                "total_registros": int(stats["total"]),
                "prom_anual": float(stats["prom_anual"]),
                "desv_anual": float(stats["desv_anual"]),
                "min_anual": float(stats["min_anual"]),
                "max_anual": float(stats["max_anual"]),
                "altitud_prom": float(stats["altitud_prom"]),
            }

            zonas = (
                self.df_spark
                .withColumn(
                    "zona_alt",
                    F.when(F.col("ALTITUD_m") < 500,  "Tierras bajas (<500 m)")
                    .when(F.col("ALTITUD_m") < 1500,  "Piso cálido (500-1500 m)")
                    .when(F.col("ALTITUD_m") < 2500,  "Piso templado (1500-2500 m)")
                    .when(F.col("ALTITUD_m") < 3500,  "Piso frío (2500-3500 m)")
                    .otherwise("Páramo (>3500 m)"),
                )
                .withColumn(
                    "orden",
                    F.when(F.col("ALTITUD_m") < 500,  1)
                    .when(F.col("ALTITUD_m") < 1500,  2)
                    .when(F.col("ALTITUD_m") < 2500,  3)
                    .when(F.col("ALTITUD_m") < 3500,  4)
                    .otherwise(5),
                )
                .groupBy("zona_alt", "orden")
                .agg(
                    F.count("*").alias("registros"),
                    F.round(F.avg("ANUAL"), 2).alias("prom_precip"),
                    F.round(F.avg("ALTITUD_m"), 0).alias("altitud_media"),
                )
                .orderBy("orden")
                .collect()
            )
            analiticas["zonas_altitudinales"] = [
                {
                    "zona": r["zona_alt"],
                    "registros": int(r["registros"]),
                    "prom_precip": float(r["prom_precip"]),
                    "altitud_media": int(r["altitud_media"]),
                }
                for r in zonas
            ]

            cols_upper = {c.upper(): c for c in self.df_spark.columns}
            col_dep = next((cols_upper[c] for c in cols_upper if "DEPART" in c), None)
            if col_dep:
                top = (
                    self.df_spark
                    .groupBy(col_dep)
                    .agg(
                        F.count("*").alias("estaciones"),
                        F.round(F.avg("ANUAL"), 2).alias("prom_precip"),
                        F.round(F.max("ANUAL"), 2).alias("max_precip"),
                    )
                    .orderBy(F.desc("prom_precip"))
                    .limit(10)
                    .collect()
                )
                analiticas["top_departamentos"] = [
                    {
                        "departamento": str(r[col_dep]),
                        "estaciones": int(r["estaciones"]),
                        "prom_precip": float(r["prom_precip"]),
                        "max_precip": float(r["max_precip"]),
                    }
                    for r in top
                ]
            else:
                analiticas["top_departamentos"] = []

            self.metricas["analiticas"] = analiticas
        except Exception as e:
            print(f"      Error consultas: {e}")
            self.metricas["analiticas"] = {}

    # =========================================================================
    # PERFILES DE CLUSTERS
    # =========================================================================

    def generar_perfiles(self):
        try:
            if "cluster_kmeans" not in self.df_pandas.columns:
                return
            feat_cols = self.metricas.get("features_usadas", COLUMNAS_BASE)
            best_k = self.metricas.get("mejor_k", 3)
            perfiles = []
            for cid in range(best_k):
                mask = self.df_pandas["cluster_kmeans"] == cid
                grupo = self.df_pandas[mask]
                n = len(grupo)
                perfil = {"cluster": cid, "registros": n}
                for feat in feat_cols:
                    if feat in grupo.columns:
                        perfil[f"mean_{feat}"] = round(float(grupo[feat].mean()), 2)
                alt = perfil.get("mean_ALTITUD_m", 0)
                precip = perfil.get("mean_ANUAL", 0)
                zone = (
                    "Alta montaña" if alt > 2500
                    else "Zona templada" if alt > 1000
                    else "Tierra baja"
                )
                rain = (
                    "muy lluvioso" if precip > 3000
                    else "lluvioso" if precip > 1500
                    else "moderado" if precip > 700
                    else "seco"
                )
                if "is_anomaly_if" in self.df_pandas.columns:
                    perfil["n_anomalias"] = int(
                        self.df_pandas.loc[mask, "is_anomaly_if"].sum()
                    )
                else:
                    perfil["n_anomalias"] = 0
                perfil["descripcion"] = f"{zone} — régimen {rain}"
                perfiles.append(perfil)
            self.extras["perfiles"] = perfiles
        except Exception as e:
            print(f"      Error perfiles: {e}")
            self.extras["perfiles"] = []

    # =========================================================================
    # ANÁLISIS EXPLORATORIO (EDA)
    # =========================================================================

    def analisis_exploratorio(self):
        try:
            feat_cols = self.metricas.get("features_usadas", COLUMNAS_BASE)
            df_num = self.df_pandas[feat_cols]
            eda_stats = []
            for col in feat_cols:
                s = df_num[col].dropna()
                eda_stats.append({
                    "feature": col,
                    "count": int(s.count()),
                    "mean": round(float(s.mean()), 2),
                    "std": round(float(s.std()), 2),
                    "min": round(float(s.min()), 2),
                    "p25": round(float(s.quantile(0.25)), 2),
                    "median": round(float(s.median()), 2),
                    "p75": round(float(s.quantile(0.75)), 2),
                    "max": round(float(s.max()), 2),
                    "skewness": round(float(s.skew()), 3),
                    "kurtosis": round(float(s.kurtosis()), 3),
                })
            corr = df_num.corr()
            top_corr = []
            for i in range(len(feat_cols)):
                for j in range(i + 1, len(feat_cols)):
                    c = float(corr.iloc[i, j])
                    if abs(c) > 0.3:
                        top_corr.append({
                            "feat1": feat_cols[i],
                            "feat2": feat_cols[j],
                            "correlation": round(c, 3),
                        })
            top_corr.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            n_out = 0
            for col in feat_cols:
                s = df_num[col].dropna()
                Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
                IQR = Q3 - Q1
                n_out += int(((s < Q1 - 1.5 * IQR) | (s > Q3 + 1.5 * IQR)).sum())
            self.extras["eda"] = {
                "estadisticas": eda_stats,
                "top_correlaciones": top_corr[:10],
                "n_outliers_iqr": n_out,
                "n_features": len(feat_cols),
            }
        except Exception as e:
            print(f"      Error EDA: {e}")
            self.extras["eda"] = {}

    # =========================================================================
    # COMPARATIVA DE MÉTODOS
    # =========================================================================

    def comparativa_clustering(self):
        try:
            comparativa = [
                {
                    "metodo": "K-Means (Spark MLlib)",
                    "tipo": "Centroide",
                    "silhouette": self.metricas.get("mejor_silhouette", 0),
                    "clusters": self.metricas.get("mejor_k", 0),
                    "ventaja": "Escalable, reproducible, distribuido",
                }
            ]
            if "dbscan" in self.extras:
                db = self.extras["dbscan"]
                comparativa.append({
                    "metodo": "DBSCAN",
                    "tipo": "Densidad",
                    "silhouette": db["silhouette"],
                    "clusters": db["n_clusters"],
                    "ventaja": f"Forma arbitraria, detecta ruido ({db['n_noise']} pts)",
                })
            if "jerarquico" in self.extras:
                h = self.extras["jerarquico"]
                comparativa.append({
                    "metodo": "Jerárquico (Ward)",
                    "tipo": "Jerárquico",
                    "silhouette": h["silhouette"],
                    "clusters": h["n_clusters"],
                    "ventaja": "Dendrograma interpretable, sin k previo",
                })
            self.extras["comparativa_clustering"] = comparativa
        except Exception as e:
            print(f"      Error comparativa: {e}")

    # =========================================================================
    # 11. VISUALIZACIONES (matplotlib)
    # =========================================================================

    def _apply_dark_style(self):
        plt.rcParams.update({
            "axes.facecolor": "#0a1628",
            "figure.facecolor": "#060d1a",
            "text.color": "#e2eaf5",
            "axes.labelcolor": "#94afc8",
            "xtick.color": "#94afc8",
            "ytick.color": "#94afc8",
            "axes.edgecolor": "#162a50",
            "grid.color": "#162a50",
            "grid.alpha": 0.5,
            "font.size": 10,
        })

    def generar_visualizaciones(self):
        print("[11/11] Generando visualizaciones matplotlib...")
        os.makedirs(IMG_DIR, exist_ok=True)
        self._apply_dark_style()
        graficos = {}

        # ── 1. Elbow + Silhouette ──────────────────────────────────────────
        try:
            metricas_k = self.metricas.get("por_k", [])
            if metricas_k:
                ks = [m["k"] for m in metricas_k]
                sils = [m["silhouette"] for m in metricas_k]
                inerts = [m["inertia"] for m in metricas_k]
                best_k = self.metricas.get("mejor_k", ks[int(np.argmax(sils))])

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                fig.patch.set_facecolor("#060d1a")

                ax1.plot(ks, inerts, "o-", color="#38bdf8", lw=2, ms=8)
                ax1.axvline(best_k, color="#f59e0b", ls="--", alpha=0.8,
                            label=f"k óptimo = {best_k}")
                ax1.set_title("Curva del Codo (Elbow Method)", color="#e2eaf5", pad=10)
                ax1.set_xlabel("Número de Clusters (k)")
                ax1.set_ylabel("Inercia (WSSSE)")
                ax1.legend(facecolor="#0a1628", labelcolor="#e2eaf5")
                ax1.grid(True, alpha=0.3)
                ax1.set_facecolor("#0a1628")
                ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

                bar_colors = ["#f59e0b" if k == best_k else "#38bdf8" for k in ks]
                ax2.bar(ks, sils, color=bar_colors, alpha=0.85, edgecolor="#162a50")
                best_idx = ks.index(best_k)
                ax2.annotate(
                    f"Mejor\n{sils[best_idx]:.3f}",
                    xy=(best_k, sils[best_idx]),
                    xytext=(best_k + 0.4, sils[best_idx] - 0.02),
                    color="#f59e0b", fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="#f59e0b"),
                )
                ax2.set_title("Silhouette Score por k", color="#e2eaf5", pad=10)
                ax2.set_xlabel("Número de Clusters (k)")
                ax2.set_ylabel("Silhouette Score")
                ax2.grid(True, alpha=0.3, axis="y")
                ax2.set_facecolor("#0a1628")
                ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

                plt.tight_layout()
                path = os.path.join(IMG_DIR, "spark_elbow.png")
                plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#060d1a")
                plt.close()
                graficos["elbow"] = path
                print(f"      ✓ spark_elbow.png")
        except Exception as e:
            print(f"      Error elbow: {e}")

        # ── 2. PCA Scree + Scatter ────────────────────────────────────────
        try:
            pca_info = self.extras.get("pca", {})
            var_exp = pca_info.get("varianza_explicada", [])
            var_acum = pca_info.get("varianza_acumulada", [])

            if var_exp and "PC1" in self.df_pandas.columns and "PC2" in self.df_pandas.columns:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
                fig.patch.set_facecolor("#060d1a")

                n_c = len(var_exp)
                comp_lbls = [f"PC{i+1}" for i in range(n_c)]
                ax1.bar(comp_lbls, [v * 100 for v in var_exp],
                        color="#38bdf8", alpha=0.8, label="Individual")
                ax1t = ax1.twinx()
                ax1t.plot(comp_lbls, [v * 100 for v in var_acum],
                          "o-", color="#f59e0b", lw=2, label="Acumulada")
                ax1t.axhline(80, color="#2dd4bf", ls="--", alpha=0.7, label="80% umbral")
                ax1t.set_ylabel("Varianza Acumulada (%)", color="#f59e0b")
                ax1t.tick_params(axis="y", colors="#f59e0b")
                ax1.set_title("Varianza Explicada por Componente PCA", color="#e2eaf5", pad=10)
                ax1.set_ylabel("Varianza Individual (%)")
                ax1.set_facecolor("#0a1628")
                ax1.grid(True, alpha=0.3, axis="y")
                lines1, lbl1 = ax1.get_legend_handles_labels()
                lines2, lbl2 = ax1t.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, lbl1 + lbl2,
                           facecolor="#0a1628", labelcolor="#e2eaf5", fontsize=8)

                if "cluster_kmeans" in self.df_pandas.columns:
                    clusters = self.df_pandas["cluster_kmeans"].values
                    for cid in sorted(set(clusters)):
                        mk = clusters == cid
                        ax2.scatter(
                            self.df_pandas.loc[mk, "PC1"],
                            self.df_pandas.loc[mk, "PC2"],
                            c=COLORS[int(cid) % len(COLORS)],
                            label=f"Cluster {cid}", alpha=0.6, s=30, edgecolors="none",
                        )
                    ax2.set_title("PCA — Clusters en Espacio Reducido", color="#e2eaf5", pad=10)
                    xlabel = f"PC1 ({var_exp[0]*100:.1f}%)" if var_exp else "PC1"
                    ylabel = f"PC2 ({var_exp[1]*100:.1f}%)" if len(var_exp) > 1 else "PC2"
                    ax2.set_xlabel(xlabel)
                    ax2.set_ylabel(ylabel)
                    ax2.legend(facecolor="#0a1628", labelcolor="#e2eaf5",
                               markerscale=1.5, fontsize=8)
                    ax2.set_facecolor("#0a1628")
                    ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                path = os.path.join(IMG_DIR, "spark_pca.png")
                plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#060d1a")
                plt.close()
                graficos["pca"] = path
                print(f"      ✓ spark_pca.png")
        except Exception as e:
            print(f"      Error PCA plot: {e}")

        # ── 3. DBSCAN scatter ────────────────────────────────────────────
        try:
            if "cluster_dbscan" in self.df_pandas.columns and "PC1" in self.df_pandas.columns:
                fig, ax = plt.subplots(figsize=(9, 6))
                fig.patch.set_facecolor("#060d1a")
                ax.set_facecolor("#0a1628")

                db_labels = self.df_pandas["cluster_dbscan"].values
                for lbl in sorted(set(db_labels)):
                    mk = db_labels == lbl
                    if lbl == -1:
                        color, name, alpha, marker = "#ef4444", "Ruido", 0.35, "x"
                    else:
                        color = COLORS[int(lbl) % len(COLORS)]
                        name, alpha, marker = f"Cluster {lbl}", 0.7, "o"
                    ax.scatter(
                        self.df_pandas.loc[mk, "PC1"],
                        self.df_pandas.loc[mk, "PC2"],
                        c=color, label=name, alpha=alpha, s=30,
                        marker=marker, edgecolors="none",
                    )
                db_info = self.extras.get("dbscan", {})
                ax.set_title(
                    f"DBSCAN — Clusters por Densidad "
                    f"(eps={db_info.get('eps',0):.3f}, minPts={db_info.get('min_samples',0)})",
                    color="#e2eaf5", pad=10,
                )
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.legend(facecolor="#0a1628", labelcolor="#e2eaf5",
                          markerscale=1.5, fontsize=8)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                path = os.path.join(IMG_DIR, "spark_dbscan.png")
                plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#060d1a")
                plt.close()
                graficos["dbscan"] = path
                print(f"      ✓ spark_dbscan.png")
        except Exception as e:
            print(f"      Error DBSCAN plot: {e}")

        # ── 4. Dendrograma ───────────────────────────────────────────────
        try:
            if self._linkage_matrix is not None:
                fig, ax = plt.subplots(figsize=(12, 5))
                fig.patch.set_facecolor("#060d1a")
                ax.set_facecolor("#0a1628")

                best_k = self.metricas.get("mejor_k", 4)
                Z = self._linkage_matrix
                threshold = Z[-(best_k - 1), 2] if len(Z) >= best_k else None
                dendrogram(
                    Z, ax=ax,
                    truncate_mode="lastp", p=30,
                    show_leaf_counts=True,
                    color_threshold=threshold,
                    above_threshold_color="#5a7a9a",
                    leaf_font_size=8,
                )
                ax.set_title(
                    f"Dendrograma Jerárquico — Ward Linkage "
                    f"(muestra n={self._dendro_n})",
                    color="#e2eaf5", pad=10,
                )
                ax.set_xlabel("Estaciones (agrupadas)")
                ax.set_ylabel("Distancia de Ward")
                if threshold:
                    ax.axhline(
                        threshold, color="#f59e0b", ls="--", alpha=0.7,
                        label=f"Corte k={best_k}",
                    )
                    ax.legend(facecolor="#0a1628", labelcolor="#e2eaf5", fontsize=8)
                for spine in ax.spines.values():
                    spine.set_color("#162a50")

                plt.tight_layout()
                path = os.path.join(IMG_DIR, "spark_dendrograma.png")
                plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#060d1a")
                plt.close()
                graficos["dendrograma"] = path
                print(f"      ✓ spark_dendrograma.png")
        except Exception as e:
            print(f"      Error dendrograma: {e}")

        # ── 5. Anomalías (IF + LOF) ──────────────────────────────────────
        try:
            if "anomaly_score_if" in self.df_pandas.columns:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                fig.patch.set_facecolor("#060d1a")

                scores_if = self.df_pandas["anomaly_score_if"].values
                is_an_if = self.df_pandas["is_anomaly_if"].values
                ax1.hist(scores_if[~is_an_if], bins=40,
                         color="#38bdf8", alpha=0.7, label="Normal")
                if is_an_if.sum() > 0:
                    ax1.hist(scores_if[is_an_if], bins=15,
                             color="#ef4444", alpha=0.85, label="Anomalía")
                thr_if = float(np.max(scores_if[is_an_if])) if is_an_if.sum() > 0 else 0
                ax1.axvline(thr_if, color="#f59e0b", ls="--",
                            label=f"Umbral ≈{thr_if:.3f}")
                ax1.set_title("Isolation Forest — Score de Anomalía",
                              color="#e2eaf5", pad=10)
                ax1.set_xlabel("Anomaly Score")
                ax1.set_ylabel("Frecuencia")
                ax1.legend(facecolor="#0a1628", labelcolor="#e2eaf5", fontsize=8)
                ax1.set_facecolor("#0a1628")
                ax1.grid(True, alpha=0.3, axis="y")

                if "anomaly_score_lof" in self.df_pandas.columns:
                    scores_lof = self.df_pandas["anomaly_score_lof"].values
                    is_an_lof = self.df_pandas["is_anomaly_lof"].values
                    ax2.hist(scores_lof[~is_an_lof], bins=40,
                             color="#2dd4bf", alpha=0.7, label="Normal")
                    if is_an_lof.sum() > 0:
                        ax2.hist(scores_lof[is_an_lof], bins=15,
                                 color="#ef4444", alpha=0.85, label="Anomalía")
                    ax2.set_title("LOF — Factor de Atípicos Locales",
                                  color="#e2eaf5", pad=10)
                    ax2.set_xlabel("Negative Outlier Factor")
                    ax2.set_ylabel("Frecuencia")
                    ax2.legend(facecolor="#0a1628", labelcolor="#e2eaf5", fontsize=8)
                    ax2.set_facecolor("#0a1628")
                    ax2.grid(True, alpha=0.3, axis="y")

                plt.tight_layout()
                path = os.path.join(IMG_DIR, "spark_anomalias.png")
                plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#060d1a")
                plt.close()
                graficos["anomalias"] = path
                print(f"      ✓ spark_anomalias.png")
        except Exception as e:
            print(f"      Error anomalías plot: {e}")

        # ── 6. Feature Importance ────────────────────────────────────────
        try:
            rf_info = self.extras.get("random_forest", {})
            feat_imp = rf_info.get("feature_importance", [])
            if feat_imp:
                top_n = min(15, len(feat_imp))
                imp_top = feat_imp[:top_n]

                fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.45)))
                fig.patch.set_facecolor("#060d1a")
                ax.set_facecolor("#0a1628")

                names = [f["feature"] for f in reversed(imp_top)]
                values = [f["importance"] for f in reversed(imp_top)]
                bar_colors = [COLORS[i % len(COLORS)] for i in range(len(names))]
                bars = ax.barh(names, values, color=bar_colors,
                               alpha=0.85, edgecolor="#162a50")
                for bar, val in zip(bars, values):
                    ax.text(
                        val + 0.002, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", ha="left",
                        color="#94afc8", fontsize=8,
                    )
                ax.set_title("Random Forest — Importancia de Features",
                             color="#e2eaf5", pad=10)
                ax.set_xlabel("Importancia (Gini)")
                ax.grid(True, alpha=0.3, axis="x")

                plt.tight_layout()
                path = os.path.join(IMG_DIR, "spark_rf_importance.png")
                plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#060d1a")
                plt.close()
                graficos["feature_importance"] = path
                print(f"      ✓ spark_rf_importance.png")
        except Exception as e:
            print(f"      Error RF importance plot: {e}")

        # ── 7. Perfiles por Cluster (Boxplot) ────────────────────────────
        try:
            feat_cols = self.metricas.get("features_usadas", COLUMNAS_BASE)
            profile_feats = [
                f for f in ["ANUAL", "ALTITUD_m", "LATITUD", "LONGITUD"]
                if f in feat_cols
            ][:4]
            best_k = self.metricas.get("mejor_k", 3)

            if profile_feats and "cluster_kmeans" in self.df_pandas.columns:
                fig, axes = plt.subplots(1, len(profile_feats),
                                         figsize=(4 * len(profile_feats), 5))
                fig.patch.set_facecolor("#060d1a")
                if len(profile_feats) == 1:
                    axes = [axes]

                for ax_i, feat in enumerate(profile_feats):
                    ax = axes[ax_i]
                    ax.set_facecolor("#0a1628")
                    bp_data = [
                        self.df_pandas.loc[
                            self.df_pandas["cluster_kmeans"] == cid, feat
                        ].dropna().values
                        for cid in range(best_k)
                    ]
                    bp = ax.boxplot(
                        bp_data, positions=range(best_k), widths=0.6,
                        patch_artist=True,
                        medianprops=dict(color="white", linewidth=2),
                        whiskerprops=dict(color="#94afc8"),
                        capprops=dict(color="#94afc8"),
                        flierprops=dict(marker=".", color="#94afc8", alpha=0.3),
                    )
                    for patch, color in zip(bp["boxes"], COLORS):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    ax.set_title(feat.replace("_", " "), color="#e2eaf5",
                                 pad=8, fontsize=9)
                    ax.set_xticks(range(best_k))
                    ax.set_xticklabels([f"C{i}" for i in range(best_k)],
                                       color="#94afc8")
                    ax.grid(True, alpha=0.3, axis="y")
                    for spine in ax.spines.values():
                        spine.set_color("#162a50")

                fig.suptitle("Perfiles por Cluster (Boxplot)",
                             color="#e2eaf5", fontsize=12)
                plt.tight_layout()
                path = os.path.join(IMG_DIR, "spark_perfiles.png")
                plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#060d1a")
                plt.close()
                graficos["cluster_profiles"] = path
                print(f"      ✓ spark_perfiles.png")
        except Exception as e:
            print(f"      Error perfiles plot: {e}")

        self.extras["graficos"] = graficos
        return graficos

    # =========================================================================
    # EXPORTAR CSV
    # =========================================================================

    def exportar_csv(self):
        try:
            cols = [
                c for c in [
                    "cluster_kmeans", "cluster_dbscan", "cluster_jerarquico",
                    "is_anomaly_if", "anomaly_score_if",
                    "is_anomaly_lof", "anomaly_score_lof",
                    "PC1", "PC2",
                ]
                if c in self.df_pandas.columns
            ]
            if cols:
                out_path = os.path.join("static", "spark_resultados.csv")
                self.df_pandas[cols].to_csv(out_path, index=False, encoding="utf-8")
                print(f"      CSV exportado: {out_path}")
        except Exception as e:
            print(f"      Error exportar CSV: {e}")

    # =========================================================================
    # LIBERAR RECURSOS
    # =========================================================================

    def detener_spark(self):
        try:
            if self.df_features is not None:
                self.df_features.unpersist()
            if self.df_spark is not None:
                self.df_spark.unpersist()
            if self.spark is not None:
                self.spark.stop()
                print("      SparkSession detenida")
        except Exception as e:
            print(f"      Error: {e}")
        finally:
            if self._tmp_csv_path:
                try:
                    os.unlink(self._tmp_csv_path)
                except OSError:
                    pass

    # =========================================================================
    # RESULTADOS PARA FLASK
    # =========================================================================

    def obtener_resultados(self):
        tiempo_total = round(time.time() - self.tiempo_inicio, 2)
        return {
            # ── compatibilidad hacia atrás ──
            "exito": True,
            "motor": (
                f"Apache Spark {self.spark.version}" if self.spark else "Apache Spark"
            ),
            "tiempo_procesamiento": tiempo_total,
            "num_registros": self.metricas.get("num_registros", 0),
            "mejor_k": self.metricas.get("mejor_k", 0),
            "mejor_silhouette": round(self.metricas.get("mejor_silhouette", 0), 4),
            "mejor_inertia": round(self.metricas.get("mejor_inertia", 0), 2),
            "tabla_metricas": self.metricas.get("tabla_metricas", []),
            "cluster_info": self.metricas.get("cluster_info", {}),
            "cluster_stats": self.metricas.get("cluster_stats", []),
            "benchmark": self.metricas.get("benchmark", []),
            "analiticas": self.metricas.get("analiticas", {}),
            "features_usadas": self.metricas.get("features_usadas", COLUMNAS_BASE),
            "archivo_fuente": ARCHIVO_DATOS,
            # ── análisis avanzados nuevos ──
            "eda": self.extras.get("eda", {}),
            "pca": self.extras.get("pca", {}),
            "dbscan": self.extras.get("dbscan", {}),
            "jerarquico": self.extras.get("jerarquico", {}),
            "anomalias": self.extras.get("anomalias", {}),
            "random_forest": self.extras.get("random_forest", {}),
            "perfiles": self.extras.get("perfiles", []),
            "comparativa_clustering": self.extras.get("comparativa_clustering", []),
            "graficos": self.extras.get("graficos", {}),
            "mensaje": (
                "Análisis de conocimiento profundo con Apache Spark completado exitosamente"
            ),
        }

    # =========================================================================
    # PIPELINE COMPLETO
    # =========================================================================

    def ejecutar(self):
        print("==== ANÁLISIS PROFUNDO — Climatología Colombia (Spark + sklearn) ====")
        try:
            if not self.cargar_excel_pandas():
                return {"exito": False, "error": "No se pudo cargar el Excel"}

            self.benchmark_configuraciones()

            if not self.iniciar_spark():
                return {"exito": False, "error": "No se pudo iniciar Spark"}
            if not self.cargar_datos():
                return {"exito": False, "error": "Error cargando datos en Spark"}
            if not self.procesar_datos():
                return {"exito": False, "error": "Error procesando features"}

            # EDA antes del clustering
            self.analisis_exploratorio()

            # Clustering principal con Spark
            if not self.clustering_kmeans(min_k=2, max_k=8):
                return {"exito": False, "error": "Error en K-Means"}

            # Reducción dimensional
            self.reduccion_pca()

            # Algoritmos sklearn
            self.clustering_dbscan()
            self.clustering_jerarquico()
            self.deteccion_anomalias()
            self.modelo_predictivo()

            # Perfiles y comparativa
            self.generar_perfiles()
            self.comparativa_clustering()

            # Consultas Spark SQL
            self.consultas_analiticas()

            # Visualizaciones
            self.generar_visualizaciones()

            # Exportar CSV
            self.exportar_csv()

            resultado = self.obtener_resultados()
            print(f"==== Pipeline completado en {resultado['tiempo_procesamiento']}s ====")
            return resultado

        except Exception as e:
            print(f"Error general: {e}")
            import traceback; traceback.print_exc()
            return {"exito": False, "error": str(e)}
        finally:
            self.detener_spark()


def ejecutar_clustering():
    """Función pública llamada desde Flask."""
    return AnalisisProfundoSpark().ejecutar()


if __name__ == "__main__":
    import json
    resultado = ejecutar_clustering()
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
