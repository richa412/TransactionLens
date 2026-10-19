-- =============================================================================
-- banking_analysis.sql
-- =============================================================================
-- Purpose : Analytical SQL queries for the Banking Transaction dataset.
-- Database: SQLite (created by the Streamlit app from the processed CSV)
-- Table   : transactions  (all columns from the featured CSV)
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. Top 10 highest-value individual transactions
-- ---------------------------------------------------------------------------
SELECT
    TransactionID,
    AccountID,
    TransactionAmount,
    TransactionDate,
    TransactionType,
    Channel,
    Location,
    risk_level
FROM transactions
ORDER BY TransactionAmount DESC
LIMIT 10;


-- ---------------------------------------------------------------------------
-- 2. Accounts with the highest number of transactions (most active)
-- ---------------------------------------------------------------------------
SELECT
    AccountID,
    COUNT(*)                     AS transaction_count,
    ROUND(SUM(TransactionAmount), 2)  AS total_amount,
    ROUND(AVG(TransactionAmount), 2)  AS avg_amount
FROM transactions
GROUP BY AccountID
ORDER BY transaction_count DESC
LIMIT 15;


-- ---------------------------------------------------------------------------
-- 3. Accounts with the highest total transaction amount
-- ---------------------------------------------------------------------------
SELECT
    AccountID,
    ROUND(SUM(TransactionAmount), 2) AS total_amount,
    COUNT(*)                         AS transaction_count,
    ROUND(AVG(TransactionAmount), 2) AS avg_amount
FROM transactions
GROUP BY AccountID
ORDER BY total_amount DESC
LIMIT 15;


-- ---------------------------------------------------------------------------
-- 4. Average transaction amount by transaction type (Credit vs Debit)
-- ---------------------------------------------------------------------------
SELECT
    TransactionType,
    COUNT(*)                          AS transaction_count,
    ROUND(AVG(TransactionAmount), 2)  AS avg_amount,
    ROUND(MIN(TransactionAmount), 2)  AS min_amount,
    ROUND(MAX(TransactionAmount), 2)  AS max_amount,
    ROUND(SUM(TransactionAmount), 2)  AS total_amount
FROM transactions
GROUP BY TransactionType
ORDER BY avg_amount DESC;


-- ---------------------------------------------------------------------------
-- 5. Channel usage frequency and total value
-- ---------------------------------------------------------------------------
SELECT
    Channel,
    COUNT(*)                          AS transaction_count,
    ROUND(SUM(TransactionAmount), 2)  AS total_amount,
    ROUND(AVG(TransactionAmount), 2)  AS avg_amount
FROM transactions
GROUP BY Channel
ORDER BY transaction_count DESC;


-- ---------------------------------------------------------------------------
-- 6. Top 15 locations by total transaction activity
-- ---------------------------------------------------------------------------
SELECT
    Location,
    COUNT(*)                          AS transaction_count,
    ROUND(SUM(TransactionAmount), 2)  AS total_amount,
    ROUND(AVG(TransactionAmount), 2)  AS avg_amount
FROM transactions
GROUP BY Location
ORDER BY total_amount DESC
LIMIT 15;


-- ---------------------------------------------------------------------------
-- 7. Monthly transaction trends (volume and value)
-- ---------------------------------------------------------------------------
SELECT
    STRFTIME('%Y-%m', TransactionDate) AS month,
    COUNT(*)                           AS transaction_count,
    ROUND(SUM(TransactionAmount), 2)   AS total_amount,
    ROUND(AVG(TransactionAmount), 2)   AS avg_amount
FROM transactions
GROUP BY month
ORDER BY month;


-- ---------------------------------------------------------------------------
-- 8. Top merchants by total transaction value
-- ---------------------------------------------------------------------------
SELECT
    MerchantID,
    COUNT(*)                          AS transaction_count,
    ROUND(SUM(TransactionAmount), 2)  AS total_amount,
    ROUND(AVG(TransactionAmount), 2)  AS avg_amount
FROM transactions
GROUP BY MerchantID
ORDER BY total_amount DESC
LIMIT 15;


-- ---------------------------------------------------------------------------
-- 9. Accounts with the highest number of HIGH-RISK / anomalous transactions
-- ---------------------------------------------------------------------------
SELECT
    AccountID,
    COUNT(*)                          AS high_risk_count,
    ROUND(SUM(TransactionAmount), 2)  AS high_risk_total_amount,
    ROUND(AVG(risk_score_pct), 2)     AS avg_risk_score
FROM transactions
WHERE risk_level = 'High'
GROUP BY AccountID
ORDER BY high_risk_count DESC
LIMIT 15;


-- ---------------------------------------------------------------------------
-- 10. Accounts showing unusually frequent transactions
--     (more than 2× the median account transaction count)
-- ---------------------------------------------------------------------------
WITH account_counts AS (
    SELECT
        AccountID,
        COUNT(*) AS tx_count
    FROM transactions
    GROUP BY AccountID
),
median_count AS (
    SELECT AVG(tx_count) AS median_tx   -- using AVG as proxy for median in SQLite
    FROM (
        SELECT tx_count
        FROM account_counts
        ORDER BY tx_count
        LIMIT 2 - (SELECT COUNT(*) FROM account_counts) % 2
        OFFSET (SELECT (COUNT(*) - 1) / 2 FROM account_counts)
    )
)
SELECT
    a.AccountID,
    a.tx_count,
    ROUND(m.median_tx, 1) AS dataset_median_tx
FROM account_counts a, median_count m
WHERE a.tx_count > 2 * m.median_tx
ORDER BY a.tx_count DESC;


-- ---------------------------------------------------------------------------
-- 11. Login attempts analysis — transactions with high login attempts
-- ---------------------------------------------------------------------------
SELECT
    LoginAttempts,
    COUNT(*)                          AS transaction_count,
    ROUND(AVG(TransactionAmount), 2)  AS avg_amount,
    ROUND(AVG(risk_score_pct), 2)     AS avg_risk_score
FROM transactions
GROUP BY LoginAttempts
ORDER BY LoginAttempts DESC;


-- ---------------------------------------------------------------------------
-- 12. Risk level summary by channel
-- ---------------------------------------------------------------------------
SELECT
    Channel,
    risk_level,
    COUNT(*)                         AS count,
    ROUND(AVG(TransactionAmount), 2) AS avg_amount
FROM transactions
GROUP BY Channel, risk_level
ORDER BY Channel, risk_level;


-- ---------------------------------------------------------------------------
-- 13. Customer occupation risk profile
-- ---------------------------------------------------------------------------
SELECT
    CustomerOccupation,
    COUNT(*)                          AS transaction_count,
    ROUND(AVG(risk_score_pct), 2)     AS avg_risk_score,
    SUM(CASE WHEN risk_level = 'High'   THEN 1 ELSE 0 END) AS high_risk_count,
    SUM(CASE WHEN risk_level = 'Medium' THEN 1 ELSE 0 END) AS medium_risk_count,
    SUM(CASE WHEN risk_level = 'Low'    THEN 1 ELSE 0 END) AS low_risk_count
FROM transactions
GROUP BY CustomerOccupation
ORDER BY avg_risk_score DESC;


-- ---------------------------------------------------------------------------
-- 14. Hourly transaction patterns (which hours see most activity)
-- ---------------------------------------------------------------------------
SELECT
    CAST(STRFTIME('%H', TransactionDate) AS INTEGER) AS hour_of_day,
    COUNT(*)                                          AS transaction_count,
    ROUND(AVG(TransactionAmount), 2)                  AS avg_amount
FROM transactions
GROUP BY hour_of_day
ORDER BY transaction_count DESC;


-- ---------------------------------------------------------------------------
-- 15. Full account risk profile (used by Account Risk Analysis page)
-- ---------------------------------------------------------------------------
SELECT
    AccountID,
    COUNT(*)                                                           AS total_transactions,
    ROUND(SUM(TransactionAmount), 2)                                   AS total_amount,
    ROUND(AVG(TransactionAmount), 2)                                   AS avg_amount,
    ROUND(MIN(TransactionAmount), 2)                                   AS min_amount,
    ROUND(MAX(TransactionAmount), 2)                                   AS max_amount,
    COUNT(DISTINCT DeviceID)                                           AS unique_devices,
    COUNT(DISTINCT MerchantID)                                         AS unique_merchants,
    COUNT(DISTINCT Location)                                           AS unique_locations,
    ROUND(AVG(LoginAttempts), 2)                                       AS avg_login_attempts,
    ROUND(AVG(risk_score_pct), 2)                                      AS avg_risk_score,
    SUM(CASE WHEN risk_level = 'High'   THEN 1 ELSE 0 END)             AS high_risk_count,
    SUM(CASE WHEN TransactionType = 'Credit' THEN 1 ELSE 0 END) * 1.0
        / COUNT(*)                                                     AS credit_ratio
FROM transactions
GROUP BY AccountID
ORDER BY avg_risk_score DESC;
