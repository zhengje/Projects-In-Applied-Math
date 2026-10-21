import './App.css';
import {useEffect, useMemo, useRef, useState} from "react";

// filter duplicates
function filterData(data) {
  if (!data || data.length === 0) return [];

  const options = new Set(data.map(item => item['title']))

  const filtered_names = [...options];
  filtered_names.sort()

  return filtered_names;
}

// Pull Recommendations from Backend
function App() {
  const [data, setData] = useState([])
  const [openModal, setOpenModal] = useState(false)
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState([]);
  const [isOpen, setIsOpen] = useState(false);
  const [backendData, setBackendData] = useState([]);

  const ref = useRef(null);

  useEffect(() => {
    fetch("/game_names").then(
      res => res.json()
    ).then(data => setData(data))
  }, []);
const options = useMemo(() => filterData(data), [data]);

// Multi-Select Search Bar for Names

  // click outside
  useEffect(() => {
    function handleClickOutside(e) {
      if (ref.current && !ref.current.contains(e.target)) {
        setIsOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () =>
      document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const filtered = useMemo(() => {
    return options
      .filter((opt) => !selected.includes(opt))
      .filter((opt) =>
        opt.toLowerCase().includes(query.toLowerCase())
      );
  }, [options, query, selected]);

  const handleSelect = (item) => {
    setSelected((prev) => [...prev, item]);
    setQuery("");
  };

  const handleRemove = (item) => {
    setSelected((prev) => prev.filter((i) => i !== item));
  };

  // Get Reccomendations from Backend
  const handleGetRecommendations = async() => {
  try {
    const response = await fetch("/game_recommendations", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ 
        body: selected
       }),
    });

    const data = await response.json();
    console.log("Backend Data: ", data);
  } catch (error) {
    console.error("Error: ", error);
  }
  }

  return (
    <div className="Recommender_System">
      
      <h1 id="App-h1"> Recommender System for Popular Steam Games (Up to Oct. 2023!)</h1>

      <div class="main-box-1">
        <div class="box-title">
          Choose from Other 2024-2026 Top Steam Game:
        </div>
        <div class="box-content">

    {/* Multi-Select Box for Game Names*/}
    <div ref={ref} style={{ width: 300 }}>
      <div
        style={{ border: "1px solid black", padding: 8 }}
        onClick={() => setIsOpen(true)}
      >
        {selected.map((item) => (
          <span key={item} style={{ marginRight: 4 }}>
            {item}
            <button onClick={() => handleRemove(item)}>❌</button>
          </span>
        ))}

        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search..."
        />
      </div>

      {isOpen && (
        <ul style={{ border: "1px solid gray", marginTop: 4 }}>
          {filtered.length > 0 ? (
            filtered.map((item) => (
              <li
                key={item}
                onClick={() => handleSelect(item)}
                style={{ cursor: "pointer" }}
              >
                {item}
              </li>
            ))
          ) : (
            <li>No options</li>
          )}
        </ul>
      )}
    </div>

    {/* Button to Initiate Getting Reccomendations */}
          <button type="button" onClick={handleGetRecommendations}>Get Recommendations!</button>
      </div>

    </div>
  </div>
  );
}

export default App;
