/* Search-relevance test set built from ML clickstream events.

Emits one row per (search_term, ecode) with impressions, clicks, position-debiased
impressions/clicks and product metadata. Candidates are restricted to products that are
currently web-active on the DSG banner and that have both a title and a default image URL.

Active-product definition follows the pattern used in ds-ecm-search-ranking-ltr
(sandbox/personalized-ltr-discovery/SQL/personalization/athlete_views_ecode.sql):
an ecode is DSG-web-active when it appears in entdata.web.dim_sku_bod_web_active
with web_chain_code = 'DSG'.

Position de-biasing mirrors the inverse-propensity weighting in sandbox/ltr_vanilla:
examination propensity per rank is the global CTR at that rank normalised to the top rank.

Params: {start_date}, {end_date}, {banner}, {channel}, {n_queries}, {max_candidates},
        {min_impressions_per_pair}, {min_clicked_ecodes_per_query}
*/

WITH impressions AS (
    SELECT
        LOWER(TRIM(m.search_event.term)) AS search_term,
        i.id                             AS ecode,
        i.num                            AS position,
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
    GROUP BY 1, 2, 3
),

clicks AS (
    SELECT
        LOWER(TRIM(m.search_event.term)) AS search_term,
        m.click_event.id                 AS ecode,
        m.click_event.num                AS position,
        COUNT(*)                         AS clicks
    FROM prod_ent_silver_db.sdsc.ml_events m
    WHERE m.event_date_short BETWEEN '{start_date}' AND '{end_date}'
      AND m.type = 'C'
      AND m.banner = '{banner}'
      AND m.channel = '{channel}'
      AND m.search_event.type = 'SRLP'
      AND m.search_event.term IS NOT NULL
      AND TRIM(m.search_event.term) <> ''
      AND m.click_event.id IS NOT NULL
    GROUP BY 1, 2, 3
),

/* DSG web-active catalog with title + image, one row per ecode */
active_products AS (
    SELECT
        w.ecode,
        MAX(w.default_ecode_image_url) AS image_url,
        MAX(w.brand_name)              AS brand_name,
        MAX(w.primary_category_name)   AS category_name,
        MAX(e.product_title)           AS product_title
    FROM entdata.web.dim_sku_bod_web_active w
    INNER JOIN prod_ml_feature_store_db.products.ecode e
        ON e.ecode = w.ecode
    WHERE w.web_chain_code = 'DSG'
      AND w.default_ecode_image_url IS NOT NULL
      AND w.default_ecode_image_url <> ''
      AND e.product_title IS NOT NULL
      AND e.product_title <> ''
      AND e.dsg_web_active = 'Y'
    GROUP BY w.ecode
),

pairs AS (
    SELECT
        i.search_term,
        i.ecode,
        i.position,
        i.impressions,
        COALESCE(c.clicks, 0) AS clicks
    FROM impressions i
    LEFT JOIN clicks c
        ON  c.search_term = i.search_term
        AND c.ecode       = i.ecode
        AND c.position    = i.position
    INNER JOIN active_products p
        ON p.ecode = i.ecode
),

position_ctr AS (
    SELECT
        position,
        SUM(clicks) / SUM(impressions) AS ctr
    FROM pairs
    GROUP BY position
    HAVING SUM(impressions) >= 500
),

propensity AS (
    SELECT
        position,
        GREATEST(
            LEAST(ctr / (SELECT ctr FROM position_ctr ORDER BY position LIMIT 1), 1.0),
            0.02
        ) AS examination_prob
    FROM position_ctr
),

pair_totals AS (
    SELECT
        p.search_term,
        p.ecode,
        SUM(p.impressions)                                    AS impressions,
        SUM(p.clicks)                                         AS clicks,
        /* IPW corrects the click, not the impression: P(click) = P(examine|rank) * P(relevant),
           so an unbiased relevance estimate is clicks / (impressions * examination_prob). */
        SUM(p.clicks / COALESCE(pr.examination_prob, 0.02))   AS ipw_clicks,
        SUM(p.position * p.impressions) / SUM(p.impressions)  AS mean_position
    FROM pairs p
    LEFT JOIN propensity pr ON pr.position = p.position
    GROUP BY p.search_term, p.ecode
    HAVING SUM(p.impressions) >= {min_impressions_per_pair}
),

qualified_terms AS (
    SELECT
        search_term,
        SUM(impressions) AS term_impressions
    FROM pair_totals
    GROUP BY search_term
    HAVING COUNT(DISTINCT CASE WHEN clicks > 0 THEN ecode END) >= {min_clicked_ecodes_per_query}
       AND COUNT(*) >= 10
    ORDER BY term_impressions DESC
    LIMIT {n_queries}
),

ranked_candidates AS (
    SELECT
        t.*,
        q.term_impressions,
        ROW_NUMBER() OVER (PARTITION BY t.search_term ORDER BY t.impressions DESC) AS candidate_rank
    FROM pair_totals t
    INNER JOIN qualified_terms q ON q.search_term = t.search_term
)

SELECT
    r.search_term,
    r.ecode,
    r.impressions,
    r.clicks,
    r.ipw_clicks,
    r.mean_position,
    r.term_impressions,
    ap.product_title,
    ap.image_url,
    ap.brand_name,
    ap.category_name
FROM ranked_candidates r
INNER JOIN active_products ap ON ap.ecode = r.ecode
WHERE r.candidate_rank <= {max_candidates}
ORDER BY r.term_impressions DESC, r.search_term, r.candidate_rank
