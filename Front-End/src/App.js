import './App.css';
import { useEffect, useMemo, useRef, useState } from "react";

// filter duplicates
function filterData(data) {
  const options = [];
  
  if (!data || data.length === 0) return options;

  for(let i = 0; i < data.length; i++) {
    const tags = data[i]["All_Tags"];
    if (tags) {
      for(let j = 0; j < tags.length; j++) {
        options.push(tags[j]);
      }
    }
  }


  let s = new Set(options);
  let filtered_options = [...s];

  return filtered_options;
}

function App() {
  const[data, setData] = useState([{}]);
  const[name, setName] = useState("");

  useEffect(() => {
    fetch("/game_tags").then(
      res => res.json()
    ).then(
        data => {
          setData(data)
        }
    )
  }, []);

const options = filterData(data)

  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState([]);
  const [isOpen, setIsOpen] = useState(false);

  const ref = useRef(null);

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

  const handleGetRecommendations = async() => {
  try {
    const response = await fetch("/games_steam_csv", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ 
        title: name,
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
      
      <h1 id="App-h1"> Recommender System for Popular Steam Games (2024-2026)</h1>

      <div class="main-box-1">
        <div class="box-title">
          Choose from Other 2024-2026 Top Steam Game:
        </div>
        <div class="box-content">
          <input onChange={(e)=>setName(e.target.value)} placeholder="Enter Game Title..." style={{padding: 5, width: 1450}}/> 
          <button type="button" onClick={handleGetRecommendations}>Get Recommendations!</button>
      </div>

      <div class="main-box-2">
        <div class="box-title">
          Suggest Your own Games:
        </div>
        <div class="box-content">
          <input placeholder="Enter Game Title..." style={{padding: 5, width: 1400}}/> 
          <input placeholder="Choose Primary Genre..." style={{padding: 5, width: 1400}}/> 
      
      <div
        style={{padding: 8}}
        onClick={() => setIsOpen(true)}
      > 

        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Choose Secondary Genres..."
        />
       
        <div> {selected.map((item) => (
          <span key={item} style={{ marginRight: 4 }} id="items">
            {item}
            <button onClick={() => handleRemove(item)} id="xbutton">❌</button>
          </span>
        ))}
        </div>
      
      </div>

      {isOpen && (
        <ul style={{ border: "1px solid grey", marginTop: 4}}>
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
      </div>
    </div>
  </div>
  );
}

export default App;
