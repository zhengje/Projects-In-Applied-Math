import './App.css';
import Modal from './Components/modal';
import {useEffect, useMemo, useRef, useState} from "react";

// filter duplicates
function filterData(data) {
  if (!data || data.length === 0) return [];

  const options = new Set(data.map(item => item['Titles']))

  const filtered_names = [...options];
  filtered_names.sort()

  return filtered_names;
}

// Pull Recommendations from Backend
function App() {
  const [header, setHeader] = useState("");
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
console.log(options)

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

  // Get Reccomendations from Backend (KNN)
  const handleGetRecommendationsKNN = async() => {
  try {
    const response = await fetch("/game_KNN", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ 
        body: selected
       }),
    });

    const data = await response.json();
    setBackendData(data)
    console.log("Backend Data: ", data);
  } catch (error) {
    console.error("Error: ", error);
  }
  }

    // Get Reccomendations from Backend (Similar Matrices)
  const handleGetRecommendationsMatSim = async() => {
  try {
    const response = await fetch("/game_MatSim", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ 
        body: selected
       }),
    });

    const data = await response.json();
    setBackendData(data)
    console.log("Backend Data: ", data);
  } catch (error) {
    console.error("Error: ", error);
  }
  }

  return (
    <div className="Recommender_System">
      
      <h1 id="App-h1"> Recommender System for Popular Steam Games (Up to Oct. 2023!)</h1>

      <div className="main-box-1">
        <div className="box-title">
          Choose from Other Top Steam Games:
        </div>
        <div className="box-content">

    {/* Multi-Select Box for Game Names*/}
    <div ref={ref} style={{ width: 1470 }}>
      <div
        style={{ border: "1px solid black", padding: 8 }}
        onClick={() => setIsOpen(true)}
      >
        {selected.map((item) => (
          <span key={item} style={{ color: "#ffff", borderStyle: "inset", borderColor: "#717072", borderRadius: "5px", marginRight: "10px", marginBottom: "20px", padding: "6px"}}>
            {item}
            <button onClick={() => handleRemove(item)}>❌</button>
          </span>
        ))}

        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search Games... (Click Outside to Close)"
          style = {{ width: 1440, backgroundColor: "#37414f", color: "#ffff", fontStyle: "italic", marginTop: "10px" }}
        />
      </div>

      {isOpen && (
        <ul style={{ border: "1px solid gray", marginTop: 4,  maxHeight: "375px", overflowY: 'auto' }}>
          {filtered.length > 0 ? (
            filtered.map((item) => (
              <li
                key={item}
                onClick={() => handleSelect(item)}
                style={{ cursor: "pointer", color: "#c4c3c0"}}
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

    {/* Buttons to Initiate Getting Reccomendations */}
          <button className="openModalBtn" onClick={() => {
            setOpenModal(true)
            handleGetRecommendationsKNN()
            setHeader("Recommendations for KNN Method")
            console.log(backendData)
          }}><span>Get KNN Recommendations!</span></button>

          <button className="openModalBtn" onClick={() => {
            setOpenModal(true)
            handleGetRecommendationsMatSim()
            setHeader("Recommendations for Matrix Similarities Method")
            console.log(backendData)
          }}><span>Get Matrix Similarities Recommendations!</span></button>

          <button className="openModalBtn" onClick={() => {
            setOpenModal(true)
            handleGetRecommendationsKNN()
            setHeader("Recommendations for Implicit Matrix Method")
            console.log(backendData)
          }}><span>Get Implicit Matrix Recommendations!</span></button>
          
      </div>
    </div>
    {openModal && <Modal closeModal={setOpenModal} data={backendData['returned_data']} header={header} />}
  </div>
  );
}

export default App;