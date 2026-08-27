/* Sensitivity probe: how do query and product counts scale with max_group_size?

Reproduces judgement_list.sql up to the group-size filter, then reports the surviving query and
product counts at several caps. Run once instead of building each variant blind.

Params: {start_date}, {end_date}, {banner}, {channel}, {term_pool}, {min_group_size}, {n_queries}
*/

WITH impressions AS (
    SELECT
        LOWER(TRIM(m.search_event.term)) AS search_term,
        i.id                             AS ecode,
        COUNT(*)                         AS impressions
    FROM prod_ent_silver_db.sdsc.ml_events m
    LATERAL VIEW INLINE(m.search_result.items) i
    WHERE m.event_date_short BETWEEN '{start_date}' AND '{end_date}'
      AND m.type = 'I'
      AND m.banner = '{banner}'
      AND m.channel = '{channel}'
      AND m.search_event.type = 'SRLP'
      AND m.search_event.page = 0
      AND m.search_event.term IS NOT NULL
      AND TRIM(m.search_event.term) <> ''
      AND i.type IN ('P', 'PP', 'SP')
    GROUP BY 1, 2
),

top_terms AS (
    SELECT search_term
    FROM impressions
    GROUP BY search_term
    ORDER BY SUM(impressions) DESC
    LIMIT {term_pool}
),

active_products AS (
    SELECT w.ecode
    FROM entdata.web.dim_sku_bod_web_active w
    INNER JOIN prod_ml_feature_store_db.products.ecode e ON e.ecode = w.ecode
    WHERE w.web_chain_code = 'DSG'
      AND w.default_ecode_image_url IS NOT NULL AND w.default_ecode_image_url <> ''
      AND e.product_title IS NOT NULL AND e.product_title <> ''
      AND e.dsg_web_active = 'Y'
    GROUP BY w.ecode
),

pairs AS (
    SELECT i.search_term, i.ecode, i.impressions
    FROM impressions i
    INNER JOIN top_terms t ON t.search_term = i.search_term
    INNER JOIN active_products p ON p.ecode = i.ecode
),

group_sizes AS (
    SELECT search_term, COUNT(*) AS n_products, SUM(impressions) AS term_impressions
    FROM pairs
    GROUP BY search_term
),

caps AS (SELECT EXPLODE(ARRAY(240, 1000, 2000, 100000)) AS cap),

ranked AS (
    SELECT
        c.cap,
        g.search_term,
        g.n_products,
        ROW_NUMBER() OVER (PARTITION BY c.cap ORDER BY g.term_impressions DESC) AS rn
    FROM group_sizes g
    CROSS JOIN caps c
    WHERE g.n_products BETWEEN {min_group_size} AND c.cap
),

selected AS (
    SELECT cap, search_term, n_products FROM ranked WHERE rn <= {n_queries}
)

SELECT
    s.cap,
    COUNT(DISTINCT s.search_term)                AS n_queries,
    SUM(s.n_products)                            AS n_pairs,
    COUNT(DISTINCT p.ecode)                      AS n_unique_products,
    ROUND(AVG(s.n_products), 1)                  AS mean_pool,
    PERCENTILE(s.n_products, 0.5)                AS median_pool,
    MAX(s.n_products)                            AS max_pool
FROM selected s
INNER JOIN pairs p ON p.search_term = s.search_term
GROUP BY s.cap
ORDER BY s.cap
