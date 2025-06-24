// pages/HomePage.js
import React, { useState } from 'react';
import axios from 'axios';


const HomePage = () => {
  const [text, setText] = useState("");
  const [sentence_number, setSentenceNumber] = useState("");
  const [result, setResult] = useState(null);
  const [feedbackuser, setFeedbackUser] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!text.trim()) {
      alert("Please enter some text first.");
      return;
    }
    // setLoading(true);
    try {
      const response = await axios.post("http://localhost:8000/api/predict/", { text ,sentence_number});
      setResult({prediction : response.data.prediction,
                tfidf : response.data.tfidf,
                HuggFace : response.data.HuggFace
     } );
    } catch (error) {
      console.error("Tahmin yapılırken hata oluştu:", error);
    }
    setLoading(false);
  };

  const resetForm = () => {
    setText("");
    setSentenceNumber("");
    setResult(null);
    setFeedbackUser("");
  };

  const handleFeedback = async () => {
    try {
      let label;

      if (feedbackuser === "True") {
        label = result.prediction;
      } else if (feedbackuser === "False") {
        label = result.prediction === 1 ? 0 : 1;
      } else {
        alert("Please select feedback.");
        return;
      }

      const response = await axios.post("http://localhost:8000/api/feedback/", {
        text: text,
        label: label
      });

      alert(response.data.message);
      resetForm();
    } catch (error) {
      console.error("Feedback eklenirken hata:", error);
      alert("Bir hata oluştu: " + error.message);
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif', maxWidth: '600px', margin: 'auto' }}>
      <h2>Self-Learning Feedback Classification System</h2>

      <textarea
        rows="5"
        cols="50"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Please enter your feedback here..."
      />
      <br />

      <textarea
        rows="2"
        cols="25"
        value={sentence_number}
        onChange={(e) => setSentenceNumber(e.target.value)}
        placeholder="the number of sentences you want in the summary"
        
      />

      <button onClick={handleSubmit} disabled={loading || !text.trim()}>
        {loading ? "Predicting..." : "Predict Sentiment"}
      </button>

      

      <h3>
        Model Prediction: {result === null ? "" : (result.prediction === 1 ? "Positive" : "Negative")}
      </h3>
    <h3>
        Summary-TFIDF: {result === null ? "" : (result.tfidf)}
      </h3>
      <h3>
        Summary-HuggFace: {result === null ? "" : (result.HuggFace)}
      </h3>
      {result !== null && (
        <div>
          <h4>Was the prediction correct?</h4>
          <select value={feedbackuser} onChange={(e) => setFeedbackUser(e.target.value)}>
            <option value="">Select</option>
            <option value="True">Correct</option>
            <option value="False">Wrong</option>
          </select>
          <button onClick={handleFeedback} disabled={!feedbackuser}>Submit Feedback</button>
        </div>
      )}
    </div>
  );
};

export default HomePage;
