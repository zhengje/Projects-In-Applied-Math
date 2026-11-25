import './App.css';
import Modal from './Components/Modal/modal';
import Table from './Components/Table/table';
import {useEffect, useMemo, useRef, useState} from "react";

// filter duplicates
function filterData(data) {
  if (!data || data.length === 0) return [];

  const options = new Set(data.map(item => item['Titles']))

  const filtered_names = [...options];
  filtered_names.sort()

  return filtered_names;
}


function App() {
  const [header, setHeader] = useState("");
  const [data, setData] = useState([])
  const [openModal, setOpenModal] = useState(false)
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState([]);
  const [isOpen, setIsOpen] = useState(false);
  const [backendData, setBackendData] = useState([]);
  const [openModels, setOpenModels] = useState(false);



  const [modelTag, setModelTag] = useState('');


// Loading Bar
  const [isLoading, setIsLoading] = useState(false);

  const ref = useRef(null);

// Pull Recommendations from Backend
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

  // Get Recommendations from Backend
  const handleGetRecommendations = async() => {
  try {
    const response = await fetch("/game_recs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ 
        body: selected
       }),
    });

    
    const data = await response.json();
    console.log(data);
    setBackendData(data);

  } catch (error) {
    console.error("Error: ", error);
  } finally {
    setOpenModels(true);
    setIsLoading(false);

  }
  }


  return (
    <div className="Recommender_System">
      
      <h1 id="App-h1"> Recommender System for Popular Steam Games</h1>

      <div className="main-box-1">
        <div className="box-title">
          Choose from Other Top Steam Games:
        </div>
        <div className="box-content">

    {/* Multi-Select Box for Game Names*/}
    <div ref={ref} style={{ width: 1000 }}>
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
          onClick={() => setOpenModels(false)}
          placeholder="Search Games... (Click Outside to Close)"
          style = {{ width: 970, backgroundColor: "#37414f", color: "#ffff", fontStyle: "italic", marginTop: "10px" }}
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

    {/* Button to Initiate Getting Recommendations */}
          <button className="searchBtn" onClick={() => {
            setIsLoading(true);
            handleGetRecommendations()
            setHeader("Load Recommendations")
          }}><span>Load Recommendations</span></button>

          {isLoading && (
            <progress id='loading' value={null} />
          )}
          
      </div>
      </div>


      {/* Modals to Display Recommendations */}
      {openModels && (
        <div className="main-box-1">
        <div className="box-title">
          View Recommendations
        </div>
        <div className="box-content">

          <button className="openModalBtn" onClick={() => {
            setOpenModal(true)
            setModelTag('KNN');
            setHeader("Recommendations for KNN Method")
          }}><span>KNN Recommendations</span></button>

          <button className="openModalBtn" onClick={() => {
            setOpenModal(true)
            setModelTag('SIM')
            setHeader("Recommendations for Matrix Similarities Method")
          }}><span>Matrix Similarities Recommendations</span></button>

          <button className="openModalBtn" onClick={() => {
            setOpenModal(true)
            setModelTag('IMF');
            setHeader("Recommendations for Implicit Matrix Method")
          }}><span>Implicit Matrix Recommendations</span></button>

          <button className="openModalBtn" onClick={() => {
            setOpenModal(true)
            setModelTag('GNN')
            setHeader("Recommendations for GNN Method")
          }}><span>GNN Recommendations</span></button>

          <button className="openModalBtn" onClick={() => {
            setOpenModal(true)
            setModelTag('JAC');
            setHeader("Recommendations for Jaccard Method")
          }}><span>Pairwise Jaccard Recommendations</span></button>
          
      </div>
    </div>
    )}
    
    {/* Table Display for Recommendation Summary */}
    {openModels &&
      <div className="main-box-1">
        <div className="box-title">
          Recommendation Summary
        </div>

        <div className="box-content">
          <Table 
          KNN={backendData.returned_data['KNN']['Titles']}
          SIM={backendData.returned_data.SIM.Titles}
          IMF={backendData.returned_data.IMF.Titles}
          GNN={backendData.returned_data.GNN.Titles}
          JAC={backendData.returned_data.JAC.Titles}
          />
        </div>
      </div>
    }

    {openModal && <Modal closeModal={setOpenModal} data={backendData.returned_data[modelTag]} header={header} />}
  </div>
  );
}

export default App;