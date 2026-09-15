import { useState } from 'react'; 
import './vendor/Bootstrap/bootstrap.min.css'; 
import { ShieldAlert, Globe, FileText, Cpu, Upload, AlertTriangle, CheckCircle, List } from 'lucide-react';
import { detectUrl, uploadCsv, generatePhishing } from './api';

function App() {
  const [activeTab, setActiveTab] = useState('single');
  const [url, setUrl] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [csvData, setCsvData] = useState(null);


  const getStats = () => {
    if (!csvData) return { total: 0, phishing: 0, legitimate: 0 };
    const total = csvData.length;
    const phishing = csvData.filter(r => r.prediction === 'phishing').length;
    const legitimate = total - phishing;
    return { total, phishing, legitimate };
  };

  const stats = getStats();

  const handleDetect = async () => {
    if(!url) return alert("Please enter a URL");
    setLoading(true);
    try {
      const res = await detectUrl(url);
      setResult(res.data);
    } catch (err) { 
      alert("Backend is offline!"); 
    }
    setLoading(false);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    setLoading(true);
    try {
      const res = await uploadCsv(formData);
      setCsvData(res.data.results);
    } catch (err) { 
      alert("Error processing CSV file"); 
    }
    setLoading(false);
  };

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const res = await generatePhishing();
      setCsvData(res.data);
      setActiveTab('bulk');
    } catch (err) {
      alert("Error generating URLs");
    }
    setLoading(false);
  };

  return (
    <div className="bg-light min-vh-100 text-dark pb-5">
      {/* Navbar */}
      <nav className="navbar navbar-dark bg-dark mb-4 shadow">
        <div className="container">
          <span className="navbar-brand d-flex align-items-center gap-2 fw-bold text-uppercase">
            <ShieldAlert className="text-warning" /> PhishGuard AI
          </span>
          <div className="d-flex gap-2">
            <button onClick={() => {setActiveTab('single'); setCsvData(null); setResult(null);}} className={`btn btn-sm ${activeTab==='single'?'btn-primary':'btn-outline-light'}`}>Individual Scan</button>
            <button onClick={() => {setActiveTab('bulk'); setCsvData(null); setResult(null);}} className={`btn btn-sm ${activeTab==='bulk'?'btn-primary':'btn-outline-light'}`}>Bulk Scan</button>
            <button onClick={() => {setActiveTab('gen'); setCsvData(null); setResult(null);}} className={`btn btn-sm ${activeTab==='gen'?'btn-primary':'btn-outline-light'}`}>AI Generator</button>
          </div>
        </div>
      </nav>

      <div className="container">
        
        {/* Tab 1: Single Scan */}
        {activeTab === 'single' && (
          <div className="card shadow-sm border-0 p-4">
            <h5 className="mb-4 d-flex align-items-center gap-2 text-primary">
              <Globe /> URL Real-time Analysis
            </h5>
            <div className="input-group mb-3">
              <input type="text" className="form-control" placeholder="Enter URL to check safety..." value={url} onChange={(e)=>setUrl(e.target.value)} />
              <button className="btn btn-primary px-4 shadow-sm" onClick={handleDetect} disabled={loading}>
                {loading ? 'Analyzing...' : 'Analyze Now'}
              </button>
            </div>
            {result && (
              <div className={`alert mt-3 d-flex align-items-center gap-3 border-2 ${result.prediction === 'phishing' ? 'alert-danger' : 'alert-success'}`}>
                {result.prediction === 'phishing' ? <AlertTriangle size={40}/> : <CheckCircle size={40}/>}
                <div>
                  <h4 className="alert-heading text-uppercase fw-bold mb-0">{result.prediction}</h4>
                  <p className="mb-0">Confidence: {(result.phishing_probability * 100).toFixed(2)}%</p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Bulk Scan & Stats Dashboard */}
        {activeTab === 'bulk' && (
          <div className="row g-4">
            {/* Stats Sidebar / Top Cards */}
            {csvData && (
              <div className="col-12">
                <div className="row g-3 mb-2">
                  <div className="col-md-4">
                    <div className="card border-0 shadow-sm bg-primary text-white p-3 d-flex flex-row align-items-center justify-content-between">
                      <div><h6 className="mb-1 text-uppercase small">Total Analyzed</h6><h3 className="mb-0 fw-bold">{stats.total}</h3></div>
                      <List size={32} opacity={0.5} />
                    </div>
                  </div>
                  <div className="col-md-4">
                    <div className="card border-0 shadow-sm bg-danger text-white p-3 d-flex flex-row align-items-center justify-content-between">
                      <div><h6 className="mb-1 text-uppercase small">Phishing Detected</h6><h3 className="mb-0 fw-bold">{stats.phishing}</h3></div>
                      <AlertTriangle size={32} opacity={0.5} />
                    </div>
                  </div>
                  <div className="col-md-4">
                    <div className="card border-0 shadow-sm bg-success text-white p-3 d-flex flex-row align-items-center justify-content-between">
                      <div><h6 className="mb-1 text-uppercase small">Legitimate Safe</h6><h3 className="mb-0 fw-bold">{stats.legitimate}</h3></div>
                      <CheckCircle size={32} opacity={0.5} />
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="col-12">
              
              <div className="card shadow-sm border-0 p-4 text-center">
                <FileText size={48} className="text-secondary mb-2 mx-auto"/> 
                <h5 className="mb-3 fw-bold">Bulk Analysis Console</h5>
                <div className="d-flex justify-content-center align-items-center gap-3 mb-4 bg-white p-3 border rounded shadow-sm mx-auto" style={{maxWidth: '500px'}}>
                    <Upload size={24} className="text-primary" />
                    <input type="file" className="form-control border-0" accept=".csv" onChange={handleFileUpload} />
                </div>
                
                {loading && (
                    <div className="my-4"><div className="spinner-border text-primary"></div><p className="text-muted mt-2">Processing dataset...</p></div>
                )}

                {csvData && (
                  <div className="table-responsive">
                    <table className="table table-hover mt-3 border bg-white rounded shadow-sm">
                      <thead className="table-dark text-uppercase small">
                        <tr><th>URL</th><th>Status</th><th>Confidence Score</th></tr>
                      </thead>
                      <tbody>
                        {csvData.map((r, i) => (
                          <tr key={i}>
                            <td className="text-start text-truncate" style={{maxWidth:'350px'}}>{r.original_url}</td>
                            <td><span className={`badge rounded-pill ${r.prediction==='phishing'?'bg-danger':'bg-success'}`}>{r.prediction}</span></td>
                            <td className="fw-bold">{(r.phishing_probability*100).toFixed(1)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Tab 3: Generator */}
        {activeTab === 'gen' && (
          <div className="card shadow-sm p-5 text-center bg-white border-0">
            <Cpu size={70} className="text-info mb-4 mx-auto animate-pulse" />
            <h3 className="fw-bold">Synthetic URL Generation</h3>
            <p className="text-muted mb-4 px-lg-5">Generate AI-powered phishing URLs to evaluate your LSTM model's performance.</p>
            <button className="btn btn-info text-white btn-lg px-5 shadow rounded-pill fw-bold" onClick={handleGenerate} disabled={loading}>
              {loading ? 'AI is Working...' : 'Generate Adversarial Samples'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;