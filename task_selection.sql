-- SWE-bench Verified task selection (METHOD.tex)
-- 25 easy (<15 min) / 35 medium (15 min–1 hour); exclude 1–4 hours and >4 hours.
-- Harder tiers are dropped because Qwen3-8B rarely succeeds there, so they
-- contribute little outcome separation and can let probes shortcut via difficulty.
-- Repo-balanced within each difficulty; global per-repo cap 8 (via 4+4).
-- Seed string '12345' is frozen; change only if you intentionally redraw.
--
-- Sanity after running:
--   SELECT difficulty, COUNT(*) FROM final_selection GROUP BY 1;
--     expect: <15 min fix=25, 15 min - 1 hour=35
--   SELECT repo, COUNT(*) FROM final_selection GROUP BY 1 ORDER BY 2 DESC;
--     expect: every count <= 8, many repos > 0

WITH filtered AS (
    SELECT *
    FROM test
    WHERE difficulty IN ('<15 min fix', '15 min - 1 hour')
),

-- Deterministic [0,1) from stable key + seed (DuckDB hash → UBIGINT).
seeded AS (
    SELECT
        *,
        (hash(repo || '|' || instance_id || '|12345')::DOUBLE
         / 18446744073709551615.0) AS rnd
    FROM filtered
),

-- Round-robin within each (difficulty, repo): 1st of every repo, then 2nd, ...
within_repo AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY difficulty, repo
            ORDER BY rnd
        ) AS rn_in_repo
    FROM seeded
),

-- Cap how many we may take from one repo *inside* each stratum
-- so that 4+4 <= 8 globally. Do this BEFORE cutting difficulty quotas
-- (otherwise django-heavy top-by-rnd slices get truncated below 60).
stratum_eligible AS (
    SELECT *
    FROM within_repo
    WHERE (difficulty = '<15 min fix'    AND rn_in_repo <= 4)  -- easy
       OR (difficulty = '15 min - 1 hour' AND rn_in_repo <= 4)  -- medium
),

-- Fill difficulty quotas in round-robin order (low rn_in_repo first).
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY difficulty
            ORDER BY rn_in_repo, rnd
        ) AS rn_diff
    FROM stratum_eligible
),

final_selection AS (
    SELECT
        repo,
        instance_id,
        difficulty,
        rnd,
        rn_in_repo,
        rn_diff
    FROM ranked
    WHERE (difficulty = '<15 min fix'    AND rn_diff <= 25)
       OR (difficulty = '15 min - 1 hour' AND rn_diff <= 35)
)

SELECT *
FROM final_selection
ORDER BY difficulty, rn_diff;
