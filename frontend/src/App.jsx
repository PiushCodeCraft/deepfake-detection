import { useEffect, useState } from "react";

function App() {
  const [status, setStatus] = useState("Checking connection...");
  const [database, setDatabase] = useState("");

  useEffect(() => {
    fetch("http://localhost:5000/api/db-test")
      .then((response) => response.json())
      .then((data) => {
        setStatus(data.message);
        setDatabase(data.database);
      })
      .catch((error) => {
        console.error(error);
        setStatus("Connection failed ❌");
      });
  }, []);

  return (
    <div style={{ padding: "40px", fontFamily: "Arial" }}>
      <h1>DeepTrace</h1>

      <h2>System Connection</h2>

      <p>
        Backend: <strong>Connected ✅</strong>
      </p>

      <p>
        PostgreSQL: <strong>{status}</strong>
      </p>

      {database && (
        <p>
          Database: <strong>{database}</strong>
        </p>
      )}
    </div>
  );
}

export default App;