import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <div className="Recommender_System">
      
      <h1> Recommender System for Popular Steam Games (2024-2026)</h1>

      <h2> Choose from Pre-existing Game</h2>
      <input placeholder="Enter popular game title"/> 
      {/* Search bar for games in csv file */}

      <h2> Input a Game Not on the List</h2>
      {/* Takes user-inputted title and some tags */}
      <input placeholder="Enter any game title"/> 
      {/* Tags: Puzzle, Platformer, Strategy */}

      {/* <div>

        <img src={logo} className="App-logo" alt="logo" />

      </div> */}

    </div>

   
  );
}

export default App;
