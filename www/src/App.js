
// npm
import React, { useState } from 'react';
import { Card } from 'react-bootstrap';

// local
import { UploadFiles } from './Upload';
import './App.css';


const App = () => {
  const APP_PREFIX = "http://127.0.0.1:5000";
  const[result, setResult] = useState("");
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
        {result ? result : null}
      </Card>

    </div>
  );
}

export default App;
