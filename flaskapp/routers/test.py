from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import os
import logging
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/test", tags=["test"])
logger = logging.getLogger(__name__)

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/")
MONGO_DB = os.getenv("MONGO_DB", "dmj")

# Lazy MongoDB client - will connect when first needed
_mongo_client = None
_mongo_db = None
_mongo_collection = None

def get_collection():
    """Lazily create MongoDB connection only when needed"""
    global _mongo_client, _mongo_db, _mongo_collection
    
    if _mongo_collection is not None:
        return _mongo_collection
    
    try:
        logger.info(f"Connecting to MongoDB at {MONGO_URI}")
        from pymongo import MongoClient
        _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Test the connection
        _mongo_client.admin.command('ping')
        _mongo_db = _mongo_client[MONGO_DB]
        _mongo_collection = _mongo_db["testdata"]
        
        # Create index on timestamp for better sorting
        _mongo_collection.create_index("timestamp")
        
        logger.info(f"MongoDB connection successful. Database: {MONGO_DB}, Collection: testdata")
        return _mongo_collection
    except Exception as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
        return None

# Pydantic models
class TestData(BaseModel):
    name: str
    email: str
    message: str
    timestamp: Optional[datetime] = None

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MongoDB Test - Insert & Display</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 30px;
        }
        
        .form-section {
            background: #f7f9fc;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            border: 1px solid #e1e8ed;
        }
        
        .form-section h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 500;
        }
        
        input, textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e8ed;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        input:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        textarea {
            resize: vertical;
            min-height: 80px;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
        }
        
        .data-section {
            margin-top: 30px;
        }
        
        .data-section h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        
        .stats {
            background: #e8f5e9;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .card {
            background: white;
            border: 1px solid #e1e8ed;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            transition: box-shadow 0.3s;
        }
        
        .card:hover {
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 8px;
        }
        
        .card-title {
            font-weight: bold;
            color: #667eea;
            font-size: 1.1em;
        }
        
        .card-time {
            color: #999;
            font-size: 0.85em;
        }
        
        .card-email {
            color: #666;
            margin-bottom: 8px;
            font-size: 0.9em;
        }
        
        .card-message {
            color: #333;
            line-height: 1.5;
            margin-top: 10px;
        }
        
        .empty {
            text-align: center;
            color: #999;
            padding: 40px;
        }
        
        .success {
            background: #d4edda;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: none;
        }
        
        .error-message {
            background: #f8d7da;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: none;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .card {
            animation: fadeIn 0.5s ease-out;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>MongoDB Test Interface</h1>
            <p>Insert test data and view it instantly - No authentication required</p>
        </div>
        
        <div class="content">
            <div id="successMsg" class="success"></div>
            <div id="errorMsg" class="error-message"></div>
            
            <div class="form-section">
                <h2>Insert New Test Data</h2>
                <form id="insertForm">
                    <div class="form-group">
                        <label>Name *</label>
                        <input type="text" id="name" name="name" required placeholder="Enter your name">
                    </div>
                    <div class="form-group">
                        <label>Email *</label>
                        <input type="email" id="email" name="email" required placeholder="Enter your email">
                    </div>
                    <div class="form-group">
                        <label>Message *</label>
                        <textarea id="message" name="message" required placeholder="Enter your message"></textarea>
                    </div>
                    <button type="submit">Insert into MongoDB</button>
                </form>
            </div>
            
            <div class="data-section">
                <h2>Stored Data (Test Collection)</h2>
                <div id="dataContainer">
                    <div class="empty">Loading data...</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        async function loadData() {
            try {
                const response = await fetch('/test/get-data');
                const data = await response.json();
                
                const container = document.getElementById('dataContainer');
                if (!response.ok) {
                    container.innerHTML = `<div class="empty">Error: ${data.detail || 'Failed to load data'}</div>`;
                    return;
                }
                if (data.length === 0) {
                    container.innerHTML = '<div class="empty">No data found. Insert some data above</div>';
                    return;
                }
                
                container.innerHTML = `
                    <div class="stats">
                        Total Records: ${data.length} | Database: dmj | Collection: testdata
                    </div>
                    ${data.map(item => `
                        <div class="card">
                            <div class="card-header">
                                <span class="card-title">${escapeHtml(item.name)}</span>
                                <span class="card-time">${new Date(item.timestamp).toLocaleString()}</span>
                            </div>
                            <div class="card-email">${escapeHtml(item.email)}</div>
                            <div class="card-message">${escapeHtml(item.message)}</div>
                        </div>
                    `).join('')}
                `;
            } catch (error) {
                console.error('Error loading data:', error);
                document.getElementById('dataContainer').innerHTML = '<div class="empty">Error loading data. Make sure MongoDB is running.</div>';
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        document.getElementById('insertForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const data = {
                name: document.getElementById('name').value,
                email: document.getElementById('email').value,
                message: document.getElementById('message').value
            };
            
            document.getElementById('successMsg').style.display = 'none';
            document.getElementById('errorMsg').style.display = 'none';
            
            try {
                const response = await fetch('/test/insert-data', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    const successMsg = document.getElementById('successMsg');
                    successMsg.textContent = result.message || 'Data inserted successfully into MongoDB';
                    successMsg.style.display = 'block';
                    setTimeout(() => {
                        successMsg.style.display = 'none';
                    }, 3000);
                    
                    document.getElementById('name').value = '';
                    document.getElementById('email').value = '';
                    document.getElementById('message').value = '';
                    
                    await loadData();
                } else {
                    const errorMsg = document.getElementById('errorMsg');
                    errorMsg.textContent = result.detail || 'Error inserting data';
                    errorMsg.style.display = 'block';
                    setTimeout(() => {
                        errorMsg.style.display = 'none';
                    }, 5000);
                }
            } catch (error) {
                console.error('Error:', error);
                const errorMsg = document.getElementById('errorMsg');
                errorMsg.textContent = 'Error connecting to server: ' + error.message;
                errorMsg.style.display = 'block';
            }
        });
        
        loadData();
        setInterval(loadData, 10000);
    </script>
</body>
</html>
"""

@router.get("/")
async def test_page():
    """Serve the test HTML page"""
    return HTMLResponse(content=HTML_TEMPLATE)

@router.post("/insert-data")
async def insert_test_data(data: TestData):
    """Insert test data into MongoDB"""
    collection = get_collection()
    if collection is None:
        raise HTTPException(status_code=503, detail="MongoDB connection failed. Please check if MongoDB is running.")
    
    try:
        document = {
            "name": data.name,
            "email": data.email,
            "message": data.message,
            "timestamp": datetime.now(),
            "created_at": datetime.now()
        }
        
        result = collection.insert_one(document)
        
        return {
            "success": True,
            "id": str(result.inserted_id),
            "message": "Data inserted successfully"
        }
    except Exception as e:
        logger.error(f"Insert error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/get-data")
async def get_test_data(limit: int = 50):
    """Retrieve all test data from MongoDB"""
    collection = get_collection()
    if collection is None:
        raise HTTPException(status_code=503, detail="MongoDB connection failed")
    
    try:
        cursor = collection.find({}).sort("timestamp", -1).limit(limit)
        data = []
        
        for doc in cursor:
            data.append({
                "id": str(doc["_id"]),
                "name": doc.get("name"),
                "email": doc.get("email"),
                "message": doc.get("message"),
                "timestamp": doc.get("timestamp", datetime.now()).isoformat()
            })
        
        return data
    except Exception as e:
        logger.error(f"Retrieve error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.delete("/delete-all")
async def delete_all_test_data():
    """Delete all test data (for cleanup)"""
    collection = get_collection()
    if collection is None:
        raise HTTPException(status_code=503, detail="MongoDB connection failed")
    
    try:
        result = collection.delete_many({})
        return {
            "success": True,
            "deleted_count": result.deleted_count
        }
    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/stats")
async def get_stats():
    """Get statistics about test collection"""
    collection = get_collection()
    if collection is None:
        raise HTTPException(status_code=503, detail="MongoDB connection failed")
    
    try:
        count = collection.count_documents({})
        latest = collection.find_one({}, sort=[("timestamp", -1)])
        
        return {
            "total_records": count,
            "database": MONGO_DB,
            "collection": "testdata",
            "mongodb_connected": True,
            "latest_record": {
                "name": latest.get("name") if latest else None,
                "timestamp": latest.get("timestamp").isoformat() if latest and latest.get("timestamp") else None
            } if latest else None
        }
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")