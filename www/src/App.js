
// npm
import React, { useState } from 'react';
import { Card } from 'react-bootstrap';

// local
import { UploadFiles } from './Upload';
import './App.css';



const App = () => {
  const APP_PREFIX = "http://127.0.0.1:5000";
  const[result, setResult] = useState(null);

  const renderResult = () => {
    return result ?
      Object.entries(result).map(([key, val]) =>
          <span key={key}>{key} : {val}</span>
          )
      : null;
  }
  return (
    <div className="App">
      <Card
        style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            width: '80%',
            margin: '0px auto',
        }}
      >
        <UploadFiles
            title="Select email(s)"
            buttonText="Predict Spam OR Ham"
            endpoint={`${APP_PREFIX}/predict`}
            responseCallback={(data) => setResult(data)}
        />
        {renderResult()}
      </Card>

    </div>
  );
}

export default App;
