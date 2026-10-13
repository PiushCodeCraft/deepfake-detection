const express = require("express");
const cors = require("cors");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const { execFile, exec, execSync } = require("child_process");
require("dotenv").config();

let pool = null;
try {
    pool = require("./db");
} catch (err) {
    console.warn("PostgreSQL module failed to load, proceeding with in-memory store:", err.message);
}

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

// Uploads directory
const UPLOADS_DIR = path.join(__dirname, "uploads");
if (!fs.existsSync(UPLOADS_DIR)) {
    fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

// Multer setup
const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, UPLOADS_DIR),
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
        cb(null, "scan-" + uniqueSuffix + path.extname(file.originalname || ".jpg"));
    }
});
const upload = multer({
    storage,
    limits: { fileSize: 15 * 1024 * 1024 } // 15 MB
});

// Paths to Python scripts and directories
const DATASET_DIR = path.resolve(__dirname, "..", "dataset");
const PREDICT_SCRIPT = path.join(DATASET_DIR, "predict_image.py");
const METRICS_FILE = path.join(DATASET_DIR, "processed", "evaluation_metrics.json");
const TEST_IMAGES_DIR = path.join(DATASET_DIR, "test_images");
const BATCH_IMAGES_DIR = path.join(DATASET_DIR, "test_images", "batch");

// In-memory fallback history
const inMemoryHistory = [];

// Detect available Python binary across Windows (py, python), Linux/macOS (python3, python)
function detectPythonCommand() {
    if (process.env.PYTHON_PATH) return process.env.PYTHON_PATH;

    const candidates = process.platform === "win32"
        ? ["py", "python", "python3"]
        : ["python3", "python", "py"];

    for (const cmd of candidates) {
        try {
            execSync(`${cmd} --version`, { stdio: "ignore", timeout: 2500 });
            return cmd;
        } catch {
            // Continue checking next candidate
        }
    }
    return process.platform === "win32" ? "python" : "python3";
}

let pythonCmd = detectPythonCommand();
console.log(`DeepTrace detected Python command: ${pythonCmd}`);

// Helper: Run Python Predictor with automatic fallback
function runPrediction(imagePath) {
    return new Promise((resolve, reject) => {
        const candidates = [pythonCmd, "py", "python", "python3"].filter(
            (val, idx, arr) => arr.indexOf(val) === idx
        );

        const tryExecute = (index) => {
            if (index >= candidates.length) {
                return reject(
                    new Error(
                        "Python executable not found. Please ensure Python is installed and accessible in system PATH."
                    )
                );
            }

            const currentCmd = candidates[index];
            const cmd = `${currentCmd} "${PREDICT_SCRIPT}" --json "${imagePath}"`;

            exec(cmd, { cwd: DATASET_DIR, timeout: 35000 }, (error, stdout, stderr) => {
                if (error && !stdout) {
                    const errStr = ((error.message || "") + " " + (stderr || "")).toLowerCase();
                    if (
                        errStr.includes("not recognized") ||
                        errStr.includes("not found") ||
                        errStr.includes("enoent")
                    ) {
                        return tryExecute(index + 1);
                    }
                    return reject(new Error(stderr || error.message));
                }

                // Cache successful command
                pythonCmd = currentCmd;

                // Extract JSON from stdout
                try {
                    const trimmed = stdout.trim();
                    const jsonStart = trimmed.indexOf("{");
                    const jsonEnd = trimmed.lastIndexOf("}");
                    if (jsonStart === -1 || jsonEnd === -1) {
                        throw new Error("No JSON found in python output: " + stdout);
                    }
                    const parsed = JSON.parse(trimmed.slice(jsonStart, jsonEnd + 1));
                    resolve(parsed);
                } catch (parseErr) {
                    reject(
                        new Error(
                            "Failed to parse ML output: " + parseErr.message + "\nStdout: " + stdout
                        )
                    );
                }
            });
        };

        tryExecute(0);
    });
}

// Helper: Generate Forensic Explanations
function generateForensicExplanation(data) {
    if (data.status !== "success") {
        return "Unable to perform forensic analysis because no facial landmarks could be isolated.";
    }

    const { prediction, confidence, symmetry_metrics, geometry_metrics, top_features } = data;
    const notes = [];

    if (prediction === "Fake") {
        notes.push(`Classified as synthetic manipulation with ${confidence}% confidence based on 22 geometric and bilateral symmetry markers.`);

        if (symmetry_metrics.eyebrow_symmetry > 0.08) {
            notes.push(`Abnormal eyebrow height asymmetry detected (${(symmetry_metrics.eyebrow_symmetry * 100).toFixed(1)}% variance), typical of GAN warping.`);
        }
        if (symmetry_metrics.eye_aspect_symmetry > 0.15) {
            notes.push(`Inter-ocular aspect ratio mismatch detected (${(symmetry_metrics.eye_aspect_symmetry * 100).toFixed(1)}% variance).`);
        }
        if (geometry_metrics.nose_width_ratio < 0.22 || geometry_metrics.nose_width_ratio > 0.35) {
            notes.push(`Atypical facial proportion: nose-to-face width ratio is ${(geometry_metrics.nose_width_ratio).toFixed(3)}.`);
        }
    } else {
        notes.push(`Verified authentic biological capture with ${confidence}% confidence.`);
        notes.push(`Consistent bilateral symmetry observed across 478 MediaPipe landmark coordinates.`);
        notes.push(`Natural ocular aspect ratio (${geometry_metrics.left_eye_aspect_ratio.toFixed(2)} left vs ${geometry_metrics.right_eye_aspect_ratio.toFixed(2)} right) conforms to human facial norms.`);
    }

    return notes.join(" ");
}

// Ensure database table exists
(async function initDb() {
    if (!pool) return;
    try {
        await pool.query(`
            CREATE TABLE IF NOT EXISTS scan_history (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255),
                prediction VARCHAR(50) NOT NULL,
                confidence NUMERIC(5, 2) NOT NULL,
                fake_probability NUMERIC(5, 2),
                real_probability NUMERIC(5, 2),
                landmarks_count INT DEFAULT 478,
                explanation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        `);
        console.log("PostgreSQL scan_history table verified.");
    } catch (err) {
        console.warn("Database initialization failed (will use memory store):", err.message);
    }
})();

// ==============================
// ROUTES
// ==============================

// Root Test
app.get("/", (req, res) => {
    res.json({
        name: "DeepTrace Forensic API",
        status: "online",
        version: "2.0.0",
        features: 22,
        pipeline: "MediaPipe (478 landmarks) -> 22 Geometry/Symmetry Features -> Random Forest"
    });
});

// System Status
app.get("/api/status", async (req, res) => {
    let dbStatus = "Disconnected";
    if (pool) {
        try {
            await pool.query("SELECT 1");
            dbStatus = "Connected";
        } catch {
            dbStatus = "Offline (In-memory fallback active)";
        }
    }

    const modelExists = fs.existsSync(path.resolve(__dirname, "..", "models", "deeptrace_random_forest.joblib"));
    const taskExists = fs.existsSync(path.resolve(__dirname, "..", "models", "face_landmarker.task"));
    const mlStatus = modelExists && taskExists
        ? "Ready (22 features)"
        : !modelExists
        ? "Missing Random Forest Model"
        : "Missing Landmarker Task";

    res.json({
        backend: "Online",
        ml_model: mlStatus,
        python_command: pythonCmd,
        database: dbStatus,
        port: PORT
    });
});

// PostgreSQL connection test
app.get("/api/db-test", async (req, res) => {
    if (!pool) {
        return res.json({ message: "In-memory mode (DB disabled)", database: "None" });
    }
    try {
        const result = await pool.query("SELECT NOW() AS current_time");
        res.json({
            message: "Database connected successfully!",
            database: process.env.DB_NAME,
            time: result.rows[0].current_time
        });
    } catch (error) {
        res.status(500).json({
            message: "Database connection failed",
            error: error.message
        });
    }
});

// Evaluation Metrics
app.get("/api/metrics", (req, res) => {
    if (fs.existsSync(METRICS_FILE)) {
        try {
            const data = JSON.parse(fs.readFileSync(METRICS_FILE, "utf-8"));
            return res.json(data);
        } catch (err) {
            return res.status(500).json({ error: "Failed to read metrics" });
        }
    }
    res.status(404).json({ error: "Metrics not yet generated" });
});

// Serve sample images statically
app.use("/api/sample-images", express.static(TEST_IMAGES_DIR));
app.use("/api/sample-images/batch", express.static(BATCH_IMAGES_DIR));

// List available samples for testing
app.get("/api/samples", (req, res) => {
    const samples = [
        { name: "real_original.jpg", label: "Real", url: "/api/sample-images/real_original.jpg" },
        { name: "fake_original.jpg", label: "Fake", url: "/api/sample-images/fake_original.jpg" }
    ];

    if (fs.existsSync(BATCH_IMAGES_DIR)) {
        const batchFiles = fs.readdirSync(BATCH_IMAGES_DIR).filter(f => f.endsWith(".jpg") || f.endsWith(".png"));
        batchFiles.forEach(file => {
            const label = file.startsWith("fake") ? "Fake" : file.startsWith("real") ? "Real" : "Edge-Case";
            samples.push({
                name: file,
                label: label,
                url: `/api/sample-images/batch/${file}`
            });
        });
    }

    res.json(samples);
});

// Main Analysis Endpoint
app.post("/api/analyze", upload.single("image"), async (req, res) => {
    let tempPath = null;
    let filename = "uploaded_image.jpg";

    try {
        // Case 1: File uploaded via multipart
        if (req.file) {
            tempPath = req.file.path;
            filename = req.file.originalname;
        }
        // Case 2: Preset sample requested
        else if (req.body.sample) {
            const sampleName = req.body.sample;
            const directPath = path.join(TEST_IMAGES_DIR, sampleName);
            const batchPath = path.join(BATCH_IMAGES_DIR, sampleName);

            if (fs.existsSync(directPath)) {
                tempPath = directPath;
            } else if (fs.existsSync(batchPath)) {
                tempPath = batchPath;
            } else {
                return res.status(400).json({ error: `Sample image '${sampleName}' not found.` });
            }
            filename = sampleName;
        }
        // Case 3: Base64 image
        else if (req.body.imageBase64) {
            const base64Data = req.body.imageBase64.replace(/^data:image\/\w+;base64,/, "");
            const buffer = Buffer.from(base64Data, "base64");
            const tempFile = "base64-" + Date.now() + ".jpg";
            tempPath = path.join(UPLOADS_DIR, tempFile);
            fs.writeFileSync(tempPath, buffer);
            filename = req.body.filename || "pasted_image.jpg";
        } else {
            return res.status(400).json({ error: "No image file, sample, or base64 provided." });
        }

        // Run prediction
        const predictionResult = await runPrediction(tempPath);

        if (predictionResult.status === "error") {
            return res.json({
                status: "error",
                message: predictionResult.message || "No face detected in the image.",
                filename
            });
        }

        // Generate forensic explanation
        const explanationText = generateForensicExplanation(predictionResult);

        const responsePayload = {
            status: "success",
            filename,
            prediction: predictionResult.prediction,
            confidence: predictionResult.confidence,
            probabilities: predictionResult.probabilities,
            landmarks_count: predictionResult.landmarks_count,
            features: predictionResult.features,
            top_features: predictionResult.top_features,
            symmetry_metrics: predictionResult.symmetry_metrics,
            geometry_metrics: predictionResult.geometry_metrics,
            explanation: explanationText,
            timestamp: new Date().toISOString()
        };

        // Log to DB or in-memory
        if (pool) {
            try {
                await pool.query(
                    `INSERT INTO scan_history (filename, prediction, confidence, fake_probability, real_probability, landmarks_count, explanation)
                     VALUES ($1, $2, $3, $4, $5, $6, $7)`,
                    [
                        filename,
                        predictionResult.prediction,
                        predictionResult.confidence,
                        predictionResult.probabilities.Fake || 0,
                        predictionResult.probabilities.Real || 0,
                        predictionResult.landmarks_count || 478,
                        explanationText
                    ]
                );
            } catch (dbErr) {
                console.warn("DB log failed, stored in memory:", dbErr.message);
                inMemoryHistory.unshift(responsePayload);
            }
        } else {
            inMemoryHistory.unshift(responsePayload);
        }

        res.json(responsePayload);

    } catch (error) {
        console.error("Analysis error:", error);
        res.status(500).json({
            status: "error",
            error: error.message || "Internal analysis failure."
        });
    } finally {
        // Clean up uploaded files in uploads dir
        if (req.file && tempPath && fs.existsSync(tempPath) && tempPath.startsWith(UPLOADS_DIR)) {
            try { fs.unlinkSync(tempPath); } catch {}
        }
        if (req.body.imageBase64 && tempPath && fs.existsSync(tempPath) && tempPath.startsWith(UPLOADS_DIR)) {
            try { fs.unlinkSync(tempPath); } catch {}
        }
    }
});

// Scan History Endpoint
app.get("/api/history", async (req, res) => {
    if (pool) {
        try {
            const { rows } = await pool.query(
                "SELECT * FROM scan_history ORDER BY created_at DESC LIMIT 20"
            );
            return res.json(rows);
        } catch {
            return res.json(inMemoryHistory.slice(0, 20));
        }
    }
    res.json(inMemoryHistory.slice(0, 20));
});

// Start Server
app.listen(PORT, () => {
    console.log(`DeepTrace backend running on http://localhost:${PORT}`);
});