// src/Home.js
import React, { useState } from 'react';
import './Home.css';
import ImageDrop from './ImageDrop';
import RunButton from './RunButton';

function Home() {
  const [animalResult, setAnimalResult] = useState(''); // State for output

  // Function to handle the "Run" button click
  const handleRun = () => {
    const animalTypes = ["Tiger", "Elephant", "Lion", "Giraffe", "Zebra", "Panda", "Kangaroo"];
    const randomAnimal = animalTypes[Math.floor(Math.random() * animalTypes.length)];
    setAnimalResult(`Detected animal: ${randomAnimal}`);
  };

  return (
    <div className="home-page">
      <h1 className="main-heading">Welcome to Animal.ai</h1>
      <p>Upload an image to detect animals using our AI technology!</p>

      <ImageDrop />
      <RunButton onClick={handleRun} />

      {/* Output Box */}
      {animalResult && (
        <div className="output-box">
          <p>{animalResult}</p>
        </div>
      )}
    </div>
  );
}

export default Home;