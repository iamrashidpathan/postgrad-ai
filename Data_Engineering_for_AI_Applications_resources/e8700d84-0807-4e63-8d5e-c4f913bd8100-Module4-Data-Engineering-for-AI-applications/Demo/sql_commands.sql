-- =====================================================================
-- Data-Centric Thinking for AI System Development — SQL Commands Reference
-- Session 4: Data Engineering for AI Applications
-- =====================================================================
-- Standalone reference of every SQL concept covered in the session.
-- These statements match the schema created and executed live in
-- Data_Centric_Thinking_for_AI_System_Development_Demos.ipynb (Demo 4).
--
-- Compatible with SQLite. To run standalone:
--   sqlite3 training.db < sql_commands.sql
-- (training.db is created by running the notebook first.)
-- =====================================================================


-- ---------------------------------------------------------------------
-- SCHEMA
-- ---------------------------------------------------------------------

DROP TABLE IF EXISTS customers;
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    city        TEXT NOT NULL
);

DROP TABLE IF EXISTS sales;
CREATE TABLE sales (
    sale_id     INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    category    TEXT NOT NULL,
    quarter     TEXT NOT NULL,
    amount      INTEGER NOT NULL,
    sale_date   TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

-- Sample seed data (illustrative — the notebook generates a fuller,
-- randomized version of this same schema with 12 customers and 60 sales).

INSERT INTO customers (customer_id, name, city) VALUES
    (1, 'Aditi Rao',      'Mumbai'),
    (2, 'Bilal Khan',     'Bengaluru'),
    (3, 'Chetan Iyer',    'Delhi'),
    (4, 'Divya Patel',    'Pune'),
    (5, 'Esha Menon',     'Hyderabad');

INSERT INTO sales (sale_id, customer_id, category, quarter, amount, sale_date) VALUES
    (1, 1, 'Electronics',     'Q4', 12000, '2026-11-02'),
    (2, 2, 'Fashion',         'Q4',  4200, '2026-10-15'),
    (3, 1, 'Home & Kitchen',  'Q3',  3100, '2026-08-20'),
    (4, 3, 'Electronics',     'Q4',  9800, '2026-12-01'),
    (5, 4, 'Books',           'Q2',   650, '2026-05-10');


-- ---------------------------------------------------------------------
-- 1. SELECT & WHERE
-- ---------------------------------------------------------------------
-- Choose specific columns, and filter rows using a condition.

SELECT name, city
FROM customers
WHERE city = 'Mumbai';


-- ---------------------------------------------------------------------
-- 2. JOIN — combining two related tables
-- ---------------------------------------------------------------------

SELECT c.name, s.category, s.amount, s.quarter
FROM customers c
JOIN sales s ON c.customer_id = s.customer_id
ORDER BY s.amount DESC
LIMIT 10;


-- ---------------------------------------------------------------------
-- 3. JOIN TYPES — INNER vs LEFT
-- ---------------------------------------------------------------------
-- INNER JOIN: only customers who have placed at least one order.

SELECT c.customer_id, c.name, COUNT(s.sale_id) AS num_orders
FROM customers c
INNER JOIN sales s ON c.customer_id = s.customer_id
GROUP BY c.customer_id, c.name;

-- LEFT JOIN: every customer, including those with zero orders.

SELECT c.customer_id, c.name, COUNT(s.sale_id) AS num_orders
FROM customers c
LEFT JOIN sales s ON c.customer_id = s.customer_id
GROUP BY c.customer_id, c.name
ORDER BY num_orders ASC;

-- RIGHT JOIN (SQLite 3.39+): equivalent to swapping table order in a LEFT JOIN.

SELECT c.customer_id, c.name, COUNT(s.sale_id) AS num_orders
FROM sales s
RIGHT JOIN customers c ON c.customer_id = s.customer_id
GROUP BY c.customer_id, c.name;


-- ---------------------------------------------------------------------
-- 4. GROUP BY & AGGREGATION FUNCTIONS
-- ---------------------------------------------------------------------
-- Revenue, order count, and average order value by category, for Q4.

SELECT category,
       COUNT(*) AS num_orders,
       SUM(amount) AS revenue,
       ROUND(AVG(amount), 2) AS avg_order_value
FROM sales
WHERE quarter = 'Q4'
GROUP BY category
ORDER BY revenue DESC;


-- ---------------------------------------------------------------------
-- 5. HAVING vs WHERE
-- ---------------------------------------------------------------------
-- WHERE filters rows before grouping; HAVING filters groups after
-- aggregation. Only keep categories whose total revenue exceeds 100000.

SELECT category, SUM(amount) AS total_revenue
FROM sales
GROUP BY category
HAVING SUM(amount) > 100000
ORDER BY total_revenue DESC;


-- ---------------------------------------------------------------------
-- 6. ORDER BY & LIMIT
-- ---------------------------------------------------------------------
-- Top 5 highest-value transactions.

SELECT c.name, s.amount, s.category, s.sale_date
FROM sales s
JOIN customers c ON c.customer_id = s.customer_id
ORDER BY s.amount DESC
LIMIT 5;


-- ---------------------------------------------------------------------
-- 7. SUBQUERIES
-- ---------------------------------------------------------------------
-- Find every customer who has placed at least one order over 10,000.

SELECT *
FROM customers
WHERE customer_id IN (
    SELECT customer_id
    FROM sales
    WHERE amount > 10000
);


-- ---------------------------------------------------------------------
-- 8. PRACTICE CHALLENGE — ANSWER
-- ---------------------------------------------------------------------
-- Find the top 3 customers by total amount spent, across all orders.

SELECT c.name, SUM(s.amount) AS total_spent
FROM customers c
JOIN sales s ON c.customer_id = s.customer_id
GROUP BY c.customer_id, c.name
ORDER BY total_spent DESC
LIMIT 3;
