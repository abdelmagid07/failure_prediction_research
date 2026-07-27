-- Freeze seed; change only if you intentionally redraw
WITH seeded AS (
    SELECT
        repo,
        instance_id,
        difficulty,
        (hash(repo || '|' || instance_id || '|200primary')::DOUBLE
         / 18446744073709551615.0) AS rnd
    FROM test
)
SELECT repo, instance_id, difficulty, rnd
FROM seeded
ORDER BY rnd
LIMIT 200;