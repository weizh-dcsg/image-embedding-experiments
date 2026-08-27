/* LTR-aligned judgement list for the embedding evaluation.

Follows the scoring conventions of ds-ecm-search-ranking-ltr:

  sandbox/ltr_vanilla/sql/inv_propensity_weights.sql  -> tau = 1 / (impressions@k / searches@k)
  sandbox/ltr_vanilla/sql/judgement_list_base.sql     -> sigmoid time decay, normalised per query
  sandbox/ltr_vanilla/sql/smoothed_ctr_bins.sql       -> relevance 0-4 from local/global smoothed
                                                         weighted-CTR quantiles
  applications/srlp-ltr/.../job.py                    -> store/channel, min_group_size,
                                                         max_group_size, days_before_today

DELIBERATE DEVIATION -- position debiasing (see papers/W4_evaluation_validity_and_systems.md):
production applies tau to impressions AND clicks, i.e. wCTR = SUM(click*tau) / SUM(impression*tau).
That is not the correction the propensity model implies. Under the examination hypothesis
P(click) = P(examine|k) * P(relevant), an unbiased relevance estimate divides the *click* by the
examination probability and leaves the impression count alone. Weighting both sides yields
r / mean(tau) -- the true relevance divided by that product's own mean inverse propensity -- which
systematically deflates products shown at deep ranks and re-introduces position bias in the same
direction it was meant to remove (measured corr(position, grade) = -0.29, barely better than the
-0.27 of raw undebiased CTR).

Here tau is applied to clicks only. The sigmoid time decay stays on both sides: it is a recency
sample weight, not a propensity correction, so the estimator is a decay-weighted IPW click rate:

    weighted_ctr = SUM(click * tau * decay) / SUM(impression * decay)

Deviation from production (documented in the experiment report): products must be web-active with
a title and an image. That is a sampling constraint for the embedding experiment, not a change to
the scoring definition.

Query sampling: every term that survives the LTR eligibility filters (min/max group size, >= 2
relevance levels) is kept -- there is no volume-based cap on the final query count. The upstream
candidate pool is the union of {term_pool_head} highest-volume terms and a deterministic random
sample of up to {tail_pool_size} lower-volume terms (>= {tail_min_impressions} raw impressions, so
they have some chance of clearing min_group_size). Without the tail sample, genuine long-tail terms
never reach the eligibility filters at all -- they'd be cut by the volume cutoff before any quality
check runs. Each surviving query is labelled head/torso/tail by its percentile rank of total
impressions among survivors ({head_pctl}, {torso_pctl} cut points), so results can be reported per
tier instead of only in aggregate.

Params: {start_date}, {end_date}, {banner}, {channel},
        {term_pool_head}, {tail_pool_size}, {tail_min_impressions}, {head_pctl}, {torso_pctl},
        {ipw_clip_position}, {decay_flatness}, {decay_midpoint},
        {alpha}, {min_group_size}, {max_group_size}
*/

WITH
/* ---------- inverse propensity weights (tau) per rank ---------- */
impression_positions AS (
    SELECT
        m.parent_id       AS search_event_id,
        MAX(i.num)        AS max_impression_position
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
    GROUP BY m.parent_id
),

search_positions AS (
    SELECT
        COALESCE(m.parent_id, m.id) AS search_event_id,
        EXPLODE(m.search_result.items.num) AS position
    FROM prod_ent_silver_db.sdsc.ml_events m
    INNER JOIN (SELECT DISTINCT search_event_id FROM impression_positions) i
        ON (CASE WHEN m.type = 'S' THEN m.id ELSE m.parent_id END) = i.search_event_id
    WHERE m.event_date_short BETWEEN '{start_date}' AND '{end_date}'
      AND m.type IN ('S', 'SPL', 'PCL', 'SPCL')
),

searches_at_k AS (
    SELECT position, COUNT(DISTINCT search_event_id) AS search_count
    FROM search_positions
    GROUP BY position
),

filled_impression_positions AS (
    SELECT s.search_event_id, s.position
    FROM search_positions s
    LEFT JOIN impression_positions i ON i.search_event_id = s.search_event_id
    WHERE s.position <= i.max_impression_position
),

impressions_at_k AS (
    SELECT position, COUNT(DISTINCT search_event_id) AS impression_count
    FROM filled_impression_positions
    GROUP BY position
),

ipw AS (
    SELECT
        s.position AS k,
        1 / (i.impression_count / s.search_count) AS tau
    FROM searches_at_k s
    LEFT JOIN impressions_at_k i ON s.position = i.position
    WHERE s.position <= (SELECT MAX(position) FROM impressions_at_k)
      AND i.impression_count > 0
),

/* ---------- judgement list base ---------- */
impressions AS (
    SELECT
        m.parent_id                      AS search_event_id,
        LOWER(TRIM(m.search_event.term)) AS search_term,
        i.num                            AS position,
        i.id                             AS ecode
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
    GROUP BY m.parent_id, m.search_event.term, i.num, i.id
),

/* Candidate term pool, applied early to bound compute. Two parts, so long-tail terms actually
   reach the eligibility filters below instead of being excluded by a volume cutoff up front:
     head_pool  -- the {term_pool_head} highest-volume terms (covers head and torso; the LTR
                   group-size and relevance-level filters below reject a large share of these,
                   and reject the highest-volume ones most often since they exceed max_group_size)
     tail_pool  -- a deterministic random sample of up to {tail_pool_size} lower-volume terms,
                   floored at {tail_min_impressions} raw impressions so they have some chance of
                   clearing min_group_size */
term_counts AS (
    SELECT search_term, COUNT(*) AS raw_impressions
    FROM impressions
    GROUP BY search_term
),

head_pool AS (
    SELECT search_term
    FROM term_counts
    ORDER BY raw_impressions DESC
    LIMIT {term_pool_head}
),

tail_pool AS (
    SELECT t.search_term
    FROM term_counts t
    LEFT JOIN head_pool h ON h.search_term = t.search_term
    WHERE h.search_term IS NULL
      AND t.raw_impressions >= {tail_min_impressions}
    ORDER BY HASH(t.search_term)
    LIMIT {tail_pool_size}
),

top_terms AS (
    SELECT search_term FROM head_pool
    UNION
    SELECT search_term FROM tail_pool
),

searches_base AS (
    SELECT
        m.event_date_short          AS search_event_date,
        COALESCE(m.parent_id, m.id) AS search_event_id,
        LOWER(TRIM(m.search_event.term)) AS search_term,
        sr.num                      AS position,
        sr.id                       AS ecode
    FROM prod_ent_silver_db.sdsc.ml_events m
    INNER JOIN (SELECT DISTINCT search_event_id FROM impressions) i
        ON (CASE WHEN m.type = 'S' THEN m.id ELSE m.parent_id END) = i.search_event_id
    LATERAL VIEW INLINE(m.search_result.items) sr
    WHERE m.event_date_short BETWEEN '{start_date}' AND '{end_date}'
      AND m.type IN ('S', 'SPL')
),

search_dates AS (
    SELECT MAX(search_event_date) AS search_event_date, search_event_id
    FROM searches_base
    GROUP BY search_event_id
),

searches AS (
    SELECT d.search_event_date, b.search_event_id, b.search_term, b.position, b.ecode
    FROM searches_base b
    LEFT JOIN search_dates d ON b.search_event_id = d.search_event_id
    INNER JOIN top_terms t ON t.search_term = b.search_term
    GROUP BY d.search_event_date, b.search_event_id, b.search_term, b.position, b.ecode
),

max_impressions AS (
    SELECT search_event_id, MAX(position) AS max_impression_position
    FROM impressions
    GROUP BY search_event_id
),

searches_and_impressions AS (
    SELECT
        s.search_event_date,
        DATE_DIFF(DAY, s.search_event_date, '{end_date}') AS search_age,
        s.search_event_id,
        s.search_term,
        s.ecode,
        s.position,
        1 AS impression
    FROM searches s
    LEFT JOIN max_impressions m ON m.search_event_id = s.search_event_id
    WHERE s.position <= m.max_impression_position
),

clicks AS (
    SELECT
        m.parent_id                      AS search_event_id,
        LOWER(TRIM(m.search_event.term)) AS search_term,
        m.click_event.id                 AS ecode,
        m.click_event.num                AS position
    FROM prod_ent_silver_db.sdsc.ml_events m
    WHERE m.event_date_short BETWEEN '{start_date}' AND '{end_date}'
      AND m.type = 'C'
      AND m.banner = '{banner}'
      AND m.channel = '{channel}'
      AND m.click_event.id IS NOT NULL
    GROUP BY m.parent_id, m.search_event.term, m.click_event.id, m.click_event.num
),

searches_impressions_clicks AS (
    SELECT
        si.search_age,
        si.search_term,
        si.ecode,
        si.position,
        si.impression,
        CASE WHEN c.ecode IS NULL THEN 0 ELSE 1 END AS click
    FROM searches_and_impressions si
    LEFT JOIN clicks c
        ON  c.search_event_id = si.search_event_id
        AND c.ecode           = si.ecode
        AND c.position        = si.position
),

/* weight = inverse propensity (rank-clipped) * sigmoid time decay */
weights AS (
    SELECT
        s.search_term,
        s.ecode,
        s.position,
        s.impression,
        s.click,
        (
            (1 + EXP(-{decay_flatness} * {decay_midpoint}))
            / (1 + EXP({decay_flatness} * (s.search_age - {decay_midpoint})))
        ) AS decay_weight,
        w.tau * (
            (1 + EXP(-{decay_flatness} * {decay_midpoint}))
            / (1 + EXP({decay_flatness} * (s.search_age - {decay_midpoint})))
        ) AS click_weight
    FROM searches_impressions_clicks s
    LEFT JOIN ipw w
        ON w.k = CASE WHEN s.position <= {ipw_clip_position}
                      THEN s.position ELSE {ipw_clip_position} END
),

query_stats AS (
    SELECT search_term, SUM(impression) AS total_impressions, SUM(decay_weight) AS total_decay_weight
    FROM weights
    GROUP BY search_term
),

/* Per-query rescale so weighted impressions stay on the same scale as raw impression counts; it is
   a constant within a query, so it cancels out of weighted_ctr and only affects the frequency
   argument passed to PERCENTILE below. */
normed_signals AS (
    SELECT
        w.search_term,
        w.ecode,
        w.position,
        w.impression,
        w.click,
        (w.decay_weight / q.total_decay_weight) * q.total_impressions AS normalized_decay_weight,
        (w.click_weight / q.total_decay_weight) * q.total_impressions AS normalized_click_weight
    FROM weights w
    LEFT JOIN query_stats q ON q.search_term = w.search_term
    WHERE w.click_weight IS NOT NULL
),

ecode_agg AS (
    SELECT
        search_term,
        ecode,
        SUM(impression * normalized_decay_weight) AS total_weighted_impressions,
        SUM(click * normalized_click_weight)      AS total_weighted_clicks,
        SUM(impression)                     AS total_impressions,
        SUM(click)                          AS total_clicks,
        SUM(position * impression) / SUM(impression) AS mean_position
    FROM normed_signals
    GROUP BY search_term, ecode
),

/* restrict to active products with title + image before scoring */
active_products AS (
    SELECT
        w.ecode,
        MAX(w.default_ecode_image_url) AS image_url,
        MAX(w.brand_name)              AS brand_name,
        MAX(w.primary_category_name)   AS category_name,
        MAX(e.product_title)           AS product_title
    FROM entdata.web.dim_sku_bod_web_active w
    INNER JOIN prod_ml_feature_store_db.products.ecode e ON e.ecode = w.ecode
    WHERE w.web_chain_code = 'DSG'
      AND w.default_ecode_image_url IS NOT NULL AND w.default_ecode_image_url <> ''
      AND e.product_title IS NOT NULL AND e.product_title <> ''
      AND e.dsg_web_active = 'Y'
    GROUP BY w.ecode
),

qualified AS (
    SELECT
        a.search_term,
        a.ecode,
        a.total_weighted_impressions,
        a.total_weighted_clicks,
        a.total_impressions,
        a.total_clicks,
        a.mean_position,
        a.total_weighted_clicks / a.total_weighted_impressions AS weighted_ctr,
        a.total_clicks / a.total_impressions                   AS ctr
    FROM ecode_agg a
    INNER JOIN active_products p ON p.ecode = a.ecode
    WHERE a.total_weighted_impressions > 0
    QUALIFY SUM(CASE WHEN a.total_weighted_clicks > 0 THEN 1 ELSE 0 END)
            OVER (PARTITION BY a.search_term) > 0
),

ctrs AS (
    SELECT
        search_term,
        weighted_ctr,
        total_weighted_impressions,
        INT(total_weighted_impressions) AS int_total_weighted_impressions
    FROM qualified
    WHERE total_weighted_clicks > 0
),

query_quantiles AS (
    SELECT
        search_term,
        SUM(total_weighted_impressions) AS n_obs,
        MIN(weighted_ctr) AS q0,
        PERCENTILE(weighted_ctr, 0.25, int_total_weighted_impressions) AS q1,
        PERCENTILE(weighted_ctr, 0.50, int_total_weighted_impressions) AS q2,
        PERCENTILE(weighted_ctr, 0.75, int_total_weighted_impressions) AS q3,
        MAX(weighted_ctr) AS q4
    FROM ctrs
    GROUP BY search_term
),

global_quantiles AS (
    SELECT
        MIN(weighted_ctr) AS q0,
        PERCENTILE(weighted_ctr, 0.25, int_total_weighted_impressions) AS q1,
        PERCENTILE(weighted_ctr, 0.50, int_total_weighted_impressions) AS q2,
        PERCENTILE(weighted_ctr, 0.75, int_total_weighted_impressions) AS q3,
        MAX(weighted_ctr) AS q4
    FROM ctrs
),

smoothed_quantiles AS (
    SELECT
        q.search_term,
        (q.n_obs * q.q1 + {alpha} * g.q1) / (q.n_obs + {alpha}) AS q1,
        (q.n_obs * q.q2 + {alpha} * g.q2) / (q.n_obs + {alpha}) AS q2,
        (q.n_obs * q.q3 + {alpha} * g.q3) / (q.n_obs + {alpha}) AS q3,
        (q.n_obs * q.q4 + {alpha} * g.q4) / (q.n_obs + {alpha}) AS q4
    FROM query_quantiles q
    CROSS JOIN global_quantiles g
),

scored AS (
    SELECT
        q.search_term,
        q.ecode,
        q.total_weighted_impressions,
        q.total_weighted_clicks,
        q.total_impressions,
        q.total_clicks,
        q.mean_position,
        q.weighted_ctr,
        q.ctr,
        CASE
            WHEN q.total_weighted_clicks = 0 THEN 0
            WHEN q.weighted_ctr > 0        AND q.weighted_ctr <= s.q1 THEN 1
            WHEN q.weighted_ctr > s.q1     AND q.weighted_ctr <= s.q2 THEN 2
            WHEN q.weighted_ctr > s.q2     AND q.weighted_ctr <= s.q3 THEN 3
            WHEN q.weighted_ctr > s.q3     AND q.weighted_ctr <= s.q4 THEN 4
            ELSE 0
        END AS relevance
    FROM qualified q
    JOIN smoothed_quantiles s USING (search_term)
),

/* groups need at least two relevance levels and an LTR-sized candidate pool */
level_filtered AS (
    SELECT s.*
    FROM scored s
    INNER JOIN (
        SELECT search_term
        FROM scored
        GROUP BY search_term
        HAVING COUNT(DISTINCT relevance) >= 2
    ) USING (search_term)
),

group_size_filtered AS (
    SELECT *
    FROM level_filtered
    QUALIFY COUNT(1) OVER (PARTITION BY search_term)
            BETWEEN {min_group_size} AND {max_group_size}
),

/* final query sample: every term that survived the LTR filters, no volume-based cap.
   Tier is a percentile rank of total impressions computed within that survivor set, not the raw
   candidate pool, so "head" here means head-of-the-evaluable-set. */
query_tiers AS (
    SELECT
        search_term,
        SUM(total_impressions)                                  AS term_impressions,
        PERCENT_RANK() OVER (ORDER BY SUM(total_impressions))    AS volume_pct_rank
    FROM group_size_filtered
    GROUP BY search_term
)

SELECT
    g.search_term,
    CASE
        WHEN t.volume_pct_rank >= {head_pctl}  THEN 'head'
        WHEN t.volume_pct_rank >= {torso_pctl} THEN 'torso'
        ELSE 'tail'
    END AS query_tier,
    t.term_impressions,
    g.ecode,
    g.relevance,
    g.weighted_ctr,
    g.ctr,
    g.mean_position,
    g.total_weighted_impressions,
    g.total_weighted_clicks,
    g.total_impressions,
    g.total_clicks,
    p.product_title,
    p.image_url,
    p.brand_name,
    p.category_name
FROM group_size_filtered g
INNER JOIN query_tiers t ON t.search_term = g.search_term
INNER JOIN active_products p ON p.ecode = g.ecode
ORDER BY query_tier, g.search_term, g.relevance DESC
