# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Parameters
 
# Parámetros (editables desde los widgets o desde la configuración del Job)
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "databricks_news")
dbutils.widgets.text("feed_url", "https://learn.microsoft.com/en-us/azure/databricks/feed.xml")
# Revisa en "Serving" qué endpoints de Foundation Models tienes disponibles y pon aquí uno
dbutils.widgets.text("model_endpoint", "system.ai.qwen35-122b-a10b")
dbutils.widgets.text("digest_days", "7")
dbutils.widgets.text("summarize_days", "30")
 
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FEED_URL = dbutils.widgets.get("feed_url")
MODEL = dbutils.widgets.get("model_endpoint")
DIGEST_DAYS = int(dbutils.widgets.get("digest_days"))
SUMMARIZE_DAYS = int(dbutils.widgets.get("summarize_days"))
 
import re
if not re.fullmatch(r"[A-Za-z0-9_\-\.]+", MODEL):
    raise ValueError(f"Nombre de endpoint no válido: {MODEL}")
 
BRONZE = f"{CATALOG}.{SCHEMA}.release_notes_bronze"
SILVER = f"{CATALOG}.{SCHEMA}.release_notes_silver"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/digests"
 
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.digests")

# COMMAND ----------

# DBTITLE 1,Bronze: ingesta del feed RSS
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
 
try:
    resp = requests.get(FEED_URL, timeout=30, headers={"User-Agent": "release-notes-digest/1.0"})
    resp.raise_for_status()
except Exception as e:
    raise RuntimeError(
        f"No se pudo descargar el feed ({e}).\n"
        "En Free Edition el acceso a internet saliente está limitado a dominios de confianza. "
        "Si falla, verifica tu identidad con LinkedIn en el workspace para habilitar el acceso saliente."
    )
 
def clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()[:4000]
 
 
def parse_date(value):
    """Acepta fechas RSS (RFC 822) y Atom (ISO 8601)."""
    if not value:
        return None
    value = value.strip()
    try:
        return parsedate_to_datetime(value)
    except Exception:
        pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
 
 
def parse_feed(xml_bytes: bytes):
    """Parser mínimo con la librería estándar: soporta RSS 2.0 y Atom."""
    root = ET.fromstring(xml_bytes)
    atom = "{http://www.w3.org/2005/Atom}"
    entries = []
 
    # RSS 2.0: <rss><channel><item>
    for item in root.iter("item"):
        entries.append({
            "id": item.findtext("guid") or item.findtext("link"),
            "title": item.findtext("title") or "",
            "link": item.findtext("link"),
            "published": item.findtext("pubDate"),
            "summary": item.findtext("description") or "",
        })
 
    # Atom: <feed><entry>
    for entry in root.iter(f"{atom}entry"):
        link_el = entry.find(f"{atom}link[@rel='alternate']")
        if link_el is None:
            link_el = entry.find(f"{atom}link")
        link = link_el.get("href") if link_el is not None else None
        entries.append({
            "id": entry.findtext(f"{atom}id") or link,
            "title": entry.findtext(f"{atom}title") or "",
            "link": link,
            "published": entry.findtext(f"{atom}published") or entry.findtext(f"{atom}updated"),
            "summary": entry.findtext(f"{atom}summary") or entry.findtext(f"{atom}content") or "",
        })
    return entries
 
 
entries = parse_feed(resp.content)
print(f"Entradas en el feed: {len(entries)}")
 
now = datetime.now(timezone.utc)
rows = [
    {
        "item_id": e["id"],
        "title": e["title"].strip(),
        "link": e["link"],
        "published_at": parse_date(e["published"]),
        "content": clean_html(e["summary"]),
        "ingested_at": now,
    }
    for e in entries
    if e["id"]
]
 
schema_ddl = "item_id STRING, title STRING, link STRING, published_at TIMESTAMP, content STRING, ingested_at TIMESTAMP"
spark.sql(f"CREATE TABLE IF NOT EXISTS {BRONZE} ({schema_ddl})")
 
if rows:
    spark.createDataFrame(rows, schema=schema_ddl).createOrReplaceTempView("feed_batch")
    result = spark.sql(f"""
        MERGE INTO {BRONZE} t
        USING feed_batch s
        ON t.item_id = s.item_id
        WHEN NOT MATCHED THEN INSERT *
    """)
    display(result)

# COMMAND ----------

# DBTITLE 1,Silver: resumen y clasificación con IA (solo entradas nuevas)
PROMPT = """Eres un ingeniero de datos senior que trabaja a diario con Azure Databricks y escribe contenido
técnico en español para otros ingenieros de datos. Te interesan especialmente la ingeniería de datos,
el rendimiento y los costes (FinOps).
 
Analiza esta release note y responde SOLO con un JSON válido, sin texto adicional, con estas claves:
 
- "resumen": 2-3 frases en español. Di qué cambia exactamente y a quién afecta. Cíñete al texto de la
  release note: no inventes beneficios ni detalles, y respeta el estado (Beta, Preview, GA, "próximamente").
  Evita frases de relleno como "mejora la eficiencia" o "mejora la colaboración" si el texto no lo justifica.
 
- "categoria": exactamente una de [Data Engineering, SQL/BI, IA/ML, Gobierno/Seguridad, Costes/FinOps, Plataforma, Otro].
  Las actualizaciones de versión o mantenimiento sin funcionalidades nuevas van en "Plataforma".
 
- "relevancia": entero de 1 a 5 para un ingeniero de datos típico en Azure Databricks, usando ESTA escala:
  5 = cambio que afecta a casi todos y exige actuar: cambio de precios o facturación, deprecación o
      cambio incompatible, o funcionalidad central de ingesta/pipelines/jobs/SQL que pasa a GA.
  4 = funcionalidad nueva útil para muchos ingenieros de datos en su trabajo diario.
  3 = útil pero para un subconjunto (una fuente concreta, un caso de uso concreto) o todavía en Beta/Preview.
  2 = nicho: perfiles de cumplimiento, clean rooms, GPUs, agentes de IA, o funciones muy especializadas.
  1 = mantenimiento sin cambios funcionales (parches de seguridad, correcciones de errores, versiones menores).
  Sé exigente: en una semana normal, la mayoría de novedades deberían ser 2 o 3, y muy pocas 5.
 
- "idea_post": un ángulo concreto para un post de LinkedIn, en una frase. Debe proponer algo práctico
  (una comparación, un antes/después, un caso de uso, un riesgo o un coste a vigilar).
  PROHIBIDO empezar con "¿Sabías que" y usar la palabra "revolucionar".
 
Reglas de idioma:
- Mantén sin traducir los nombres de producto y las siglas: ADBC, CDC, GA, Beta, Preview, Unity Catalog,
  Lakeflow, Lakebase, Delta, Structured Streaming, Clean Rooms, Feature Views, Databricks Apps, Unity Gateway.
- Glosario: "ingress" = "entrada" (nunca "ingesta"); "egress" = "salida"; "ingestion" = "ingesta";
  "workspace" = "workspace"; "endpoint" = "endpoint"; "shuffle partitions" = "particiones de shuffle".
- Revisa la ortografía antes de responder.
 
Release note:
"""
 
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {SILVER} (
        item_id STRING, title STRING, link STRING, published_at TIMESTAMP,
        resumen STRING, categoria STRING, relevancia INT, idea_post STRING,
        raw_response STRING, processed_at TIMESTAMP
    )
""")
 
# La misma novedad aparece a veces en varias páginas (p. ej. la mensual y la del producto) con
# distinta URL pero el mismo título: nos quedamos con una sola por título.
pending = spark.sql(f"""
    WITH candidatas AS (
        SELECT b.*,
               row_number() OVER (PARTITION BY lower(trim(b.title))
                                  ORDER BY b.published_at DESC NULLS LAST, b.item_id) AS rn
        FROM {BRONZE} b
        LEFT ANTI JOIN {SILVER} s
          ON b.item_id = s.item_id OR lower(trim(b.title)) = lower(trim(s.title))
         WHERE coalesce(b.published_at, b.ingested_at) >= current_timestamp() - INTERVAL {SUMMARIZE_DAYS} DAYS
    )
    SELECT * EXCEPT (rn) FROM candidatas WHERE rn = 1
""")
n_pending = pending.count()
print(f"Entradas pendientes de resumir: {n_pending}")
 
if n_pending > 0:
    # Repartir en varias tareas para que las llamadas al modelo vayan en paralelo
    pending.repartition(4).createOrReplaceTempView("pending")
    spark.sql(
        f"""
        WITH llm AS (
            SELECT *,
                   ai_query('{MODEL}',
                            concat(:prompt, 'Título: ', title, '\nContenido: ', coalesce(content, ''))) AS raw_response
            FROM pending
        ),
        parsed AS (
            SELECT *,
                   from_json(regexp_extract(raw_response, '(?s)\\\\{{.*\\\\}}', 0),
                             'resumen STRING, categoria STRING, relevancia INT, idea_post STRING') AS j
            FROM llm
        )
        MERGE INTO {SILVER} t
        USING (
            SELECT item_id, title, link, published_at,
                   j.resumen,
                   -- Normaliza la categoría a la lista cerrada (p. ej. "AI/ML" -> "IA/ML")
                   CASE
                       WHEN lower(j.categoria) IN ('ia/ml', 'ai/ml', 'ia', 'ai', 'ml') THEN 'IA/ML'
                       WHEN lower(j.categoria) IN ('data engineering', 'ingeniería de datos') THEN 'Data Engineering'
                       WHEN lower(j.categoria) IN ('sql/bi', 'sql', 'bi') THEN 'SQL/BI'
                       WHEN lower(j.categoria) IN ('gobierno/seguridad', 'gobierno', 'seguridad', 'governance/security') THEN 'Gobierno/Seguridad'
                       WHEN lower(j.categoria) IN ('costes/finops', 'costes', 'finops', 'costs/finops') THEN 'Costes/FinOps'
                       WHEN lower(j.categoria) IN ('plataforma', 'platform') THEN 'Plataforma'
                       ELSE 'Otro'
                   END AS categoria,
                   j.relevancia, j.idea_post,
                   raw_response, current_timestamp() AS processed_at
            FROM parsed
        ) s
        ON t.item_id = s.item_id
        WHEN NOT MATCHED THEN INSERT *
        """,
        args={"prompt": PROMPT},
    )
 
display(spark.sql(f"SELECT title, categoria, relevancia, resumen FROM {SILVER} ORDER BY processed_at DESC LIMIT 20"))

# COMMAND ----------

# DBTITLE 1,Gold: digest semanal en Markdown
digest_rows = spark.sql(f"""
    SELECT title, link, categoria, relevancia, resumen, idea_post, published_at
    FROM {SILVER}
    WHERE coalesce(published_at, processed_at) >= current_timestamp() - INTERVAL {DIGEST_DAYS} DAYS
    QUALIFY row_number() OVER (PARTITION BY lower(trim(title)) ORDER BY processed_at DESC) = 1
    ORDER BY relevancia DESC NULLS LAST, published_at DESC
""").collect()
 
today = now.strftime("%Y-%m-%d")
lines = [f"# Novedades de Databricks — semana hasta {today}", ""]
 
if not digest_rows:
    lines.append("Sin novedades en el periodo.")
else:
    lines.append(f"{len(digest_rows)} novedades, ordenadas por relevancia.\n")
    for r in digest_rows:
        estrellas = "★" * (r.relevancia or 0)
        lines += [
            f"## {r.title}",
            f"*{r.categoria or 'Sin categoría'}* · {estrellas}",
            "",
            r.resumen or "_(sin resumen)_",
            "",
            f"💡 **Idea de post:** {r.idea_post or '-'}",
            f"🔗 {r.link}",
            "",
        ]
 
digest_md = "\n".join(lines)
digest_file = f"{VOLUME_PATH}/digest_{today}.md"
with open(digest_file, "w", encoding="utf-8") as f:
    f.write(digest_md)
 
print(f"Digest guardado en {digest_file}")
displayHTML(f"<pre style='white-space:pre-wrap'>{digest_md}</pre>")

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE VIEW {CATALOG}.{SCHEMA}.v_novedades_por_mes AS
    SELECT date_trunc('month', coalesce(published_at, processed_at)) AS mes,
           categoria,
           count(*) AS novedades,
           round(avg(relevancia), 2) AS relevancia_media
    FROM {SILVER}
    GROUP BY ALL
""")