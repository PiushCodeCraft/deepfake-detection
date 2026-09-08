const express = require("express");
const cors = require("cors");
require("dotenv").config();

const pool = require("./db");

const app = express();

app.use(cors());
app.use(express.json());

// Test route
app.get("/", (req, res) => {
    res.json({
        message: "DeepTrace backend is running"
    });
});

// PostgreSQL connection test
app.get("/api/db-test", async (req, res) => {
    try {
        const result = await pool.query("SELECT NOW() AS current_time");

        res.json({
            message: "Database connected successfully!",
            database: process.env.DB_NAME,
            time: result.rows[0].current_time
        });
    } catch (error) {
        console.error("Database connection error:", error.message);

        res.status(500).json({
            message: "Database connection failed",
            error: error.message
        });
    }
});

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
    console.log(`DeepTrace backend running on http://localhost:${PORT}`);
});