import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import Modal from './Modal'; // Importujeme komponentu modálneho okna

function App() {
  const [yoloModels, setYoloModels] = useState([]);
  const [imitationModels, setImitationModels] = useState([]);
  const [selectedYolo, setSelectedYolo] = useState('');
  const [selectedImitation, setSelectedImitation] = useState('');
  const [file, setFile] = useState(null);
  const [processedImage, setProcessedImage] = useState('');
  const [trajectoryImage, setTrajectoryImage] = useState('');
  const [detectedString, setDetectedString] = useState('');
  const [expectedString, setExpectedString] = useState('');
  const [comparisonResults, setComparisonResults] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  // Stavy pre modálne okno
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalImageSrc, setModalImageSrc] = useState('');

  useEffect(() => {
    axios.get('http://localhost:5000/api/models')
      .then(response => {
        setYoloModels(response.data.yolo_models);
        setImitationModels(response.data.imitation_models);
        setSelectedYolo(response.data.yolo_models[0] || '');
        setSelectedImitation(response.data.imitation_models[0] || '');
      })
      .catch(error => {
        console.error('Error fetching models:', error);
        setError('Error fetching models. Make sure the backend is running.');
      });
  }, []);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setError('');
    setProcessedImage('');
    setTrajectoryImage('');
    setDetectedString('');
    setComparisonResults(null);
    setLoading(true);

    if (!file) {
      setError('Please select a file.');
      setLoading(false);
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('yolo_model', selectedYolo);
    formData.append('imitation_model', selectedImitation);
    if (expectedString) {
      formData.append('expected_string', expectedString);
    }

    axios.post('http://localhost:5000/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    .then(response => {
      const backendUrl = 'http://localhost:5000';
      setProcessedImage(backendUrl + response.data.processed_image_url);
      setTrajectoryImage(backendUrl + response.data.trajectory_image_url);
      setDetectedString(response.data.detected_string);
      if (response.data.expected_string) {
        setComparisonResults({
          expected: response.data.expected_string,
          distance: response.data.levenshtein_distance,
          accuracy: response.data.accuracy,
          diffHtml: response.data.diff_html
        });
      }
    })
    .catch(error => {
      console.error('Error uploading file:', error);
      setError('Error processing the image. Please check the console for details.');
    })
    .finally(() => {
      setLoading(false);
    });
  };

  const openModal = (src) => {
    setModalImageSrc(src);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setModalImageSrc('');
  };

  const handleDownload = () => {
    const blob = new Blob([detectedString], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'detected_string.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="App">
      <h1>Image Processing App</h1>
      {error && <p className="error">{error}</p>}
      <form onSubmit={handleSubmit}>
        <div>
          <label>YOLO Model:</label>
          <select value={selectedYolo} onChange={(e) => setSelectedYolo(e.target.value)} disabled={loading}>
            {yoloModels.map(model => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        </div>
        <div>
          <label>Imitation Model:</label>
          <select value={selectedImitation} onChange={(e) => setSelectedImitation(e.target.value)} disabled={loading}>
            {imitationModels.map(model => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        </div>
        <div>
          <label>Upload Image:</label>
          <input type="file" onChange={handleFileChange} disabled={loading} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <label style={{ marginBottom: '5px' }}>Expected String (Optional):</label>
          <textarea 
            style={{ width: '100%', boxSizing: 'border-box', marginTop: '5px' }}
            rows="3"
            value={expectedString} 
            onChange={(e) => setExpectedString(e.target.value)} 
            disabled={loading} 
            placeholder="Enter expected string here" 
          />
        </div>
        <button type="submit" disabled={loading} style={{ marginTop: '10px' }}>
          {loading ? 'Processing...' : 'Process Image'}
        </button>
      </form>

      {loading && <div className="loader"></div>}

      <div className="results-container">
        {processedImage && (
          <div className="result-item">
            <h2>Processed Image</h2>
            <img 
              src={processedImage} 
              alt="Processed" 
              onClick={() => openModal(processedImage)}
              className="result-image"
            />
          </div>
        )}
        {trajectoryImage && (
          <div className="result-item">
            <h2>Trajectory Image</h2>
            <img 
              src={trajectoryImage} 
              alt="Trajectory" 
              onClick={() => openModal(trajectoryImage)}
              className="result-image"
            />
          </div>
        )}
      </div>

      {detectedString && (
        <div className="result-item-string">
          <div className="title-container">
            <h2>Detected String</h2>
            <button onClick={handleDownload} className="download-btn">Download .txt</button>
          </div>
          <div className="detected-string-container">
            <p>{detectedString}</p>
          </div>
          {comparisonResults && (
            <div className="comparison-container" style={{ marginTop: '20px', padding: '15px', border: '1px solid #ccc', borderRadius: '5px' }}>
              <h3>Comparison Results</h3>
              <p><strong>Levenshtein Distance:</strong> {comparisonResults.distance}</p>
              <p><strong>Accuracy:</strong> {comparisonResults.accuracy}%</p>
              {comparisonResults.diffHtml && (
                <div 
                  className="diff-container"
                  style={{ marginTop: '15px', overflowX: 'auto' }}
                  dangerouslySetInnerHTML={{ __html: comparisonResults.diffHtml }}
                />
              )}
            </div>
          )}
        </div>
      )}

      <Modal 
        isOpen={isModalOpen} 
        onClose={closeModal} 
        imageSrc={modalImageSrc} 
      />
    </div>
  );
}

export default App;
