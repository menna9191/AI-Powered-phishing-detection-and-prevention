import axios from 'axios';
const API_BASE_URL = 'http://localhost:5000'; 

export const detectUrl = (url) => axios.post(`${API_BASE_URL}/detect`, { url });
export const uploadCsv = (formData) => axios.post(`${API_BASE_URL}/upload_csv`, formData);
export const generatePhishing = () => axios.get(`${API_BASE_URL}/generate_phishing`);