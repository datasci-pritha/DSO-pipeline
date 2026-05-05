DELETE FROM test.ratings_copy
WHERE ctid NOT IN (
    SELECT MIN(ctid)
    FROM test.ratings_copy
    GROUP BY  segment, dataset, rmse, r_squared, mae
);